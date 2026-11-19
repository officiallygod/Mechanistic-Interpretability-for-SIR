"""Batched policy server thread.

Runs in the main process (so it directly owns the GPU `method`). Pulls
PredictRequests from a shared request queue, batches them with a short timeout,
runs one forward pass, and routes per-env action chunks back to each worker's
reply queue.
"""
import logging
import queue as _queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from torch_geometric.data import Batch, Data

from utils.node_scores import NodeScoreTracker
from envs.robocasa.utils import eval_noise
from envs.robocasa.parallel.protocol import (
    ActionResponse,
    PredictRequest,
    Shutdown,
)

log = logging.getLogger(__name__)


class PolicyServer:
    """Server thread that owns the GPU policy.

    Parameters
    ----------
    method : the loaded `Diffusion` method (already on GPU).
    request_q : shared mp.Queue, all workers put PredictRequests here.
    reply_qs : list of mp.Queues, one per worker (indexed by env_id).
    num_envs : number of workers feeding this server.
    batch_timeout_ms : how long to wait for additional requests after the first.
    max_eval_batch : safety cap on batch size (independent of num_envs).
    y_bounds : action bounds for clipping (sent back already clipped).
    deterministic_noise : start each prediction's diffusion from noise keyed on
        (task, rollout_idx, step) instead of the global RNG. See `eval_noise`.
    noise_seed : seed mixed into that key.
    """

    def __init__(self,
                 method,
                 request_q,
                 reply_qs,
                 num_envs: int,
                 batch_timeout_ms: float = 20.0,
                 max_eval_batch: int = 16,
                 y_bounds: Optional[torch.Tensor] = None,
                 deterministic_noise: bool = False,
                 noise_seed: int = 42,
                 ):
        self.deterministic_noise = bool(deterministic_noise)
        self.noise_seed = int(noise_seed)
        self.method = method
        self.request_q = request_q
        self.reply_qs = reply_qs
        self.num_envs = num_envs
        self.batch_timeout_ms = batch_timeout_ms / 1000.0
        self.max_eval_batch = max(1, int(max_eval_batch))
        self.y_bounds = y_bounds

        # Cached CLIP language embeddings keyed by (env_id, episode_id).
        # Avoids re-encoding the goal at every step within an episode.
        self._lang_cache: Dict[tuple, torch.Tensor] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # See _put_reply: bounding this is what stops one stuck worker from
        # deadlocking the entire evaluation.
        self.reply_put_timeout_s = 60.0
        self._dropped_replies = 0
        # A failed forward is answered with zero actions so the workers do not hang.
        # That keeps the run alive but silently degrades whatever rollouts were in
        # flight, so it has to be counted and surfaced — one ERROR line inside a
        # multi-million-line log is not something anyone finds before trusting the
        # numbers. Read back by the tester at the end of the round.
        self.zero_action_replies = 0
        self.zero_action_first_error = None
        # Rollout-time node scores. The server owns every forward pass, so this is the
        # one place that sees the whole evaluation. Worth having separately from the
        # train/val snapshots: the scorer runs on live segmentation here, and whether
        # its ranking survives that shift is not something offline data can answer.
        self.node_scores = NodeScoreTracker()

        # Throughput stats (printed periodically).
        self._stats_batches = 0
        self._stats_items = 0
        self._stats_fwd_time_s = 0.0
        self._stats_last_emit = time.monotonic()
        self._stats_emit_every = 50  # batches

    # ----- public lifecycle ----------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="PolicyServer", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        # Drain a sentinel so a blocking get() wakes up.
        try:
            self.request_q.put_nowait(Shutdown())
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._lang_cache.clear()

    # ----- main loop ------------------------------------------------------

    def _run(self):
        log.info(f"[PolicyServer] thread started (num_envs={self.num_envs}, "
                 f"max_batch={self.max_eval_batch}, timeout_ms={self.batch_timeout_ms*1000:.0f})")
        try:
            while not self._stop_event.is_set():
                # 1) block-get the first request
                try:
                    first = self.request_q.get(timeout=0.5)
                except _queue.Empty:
                    continue

                if isinstance(first, Shutdown) or getattr(first, "kind", None) == "shutdown":
                    break
                if not isinstance(first, PredictRequest):
                    continue

                batch: List[PredictRequest] = [first]
                deadline = time.monotonic() + self.batch_timeout_ms
                # 2) drain extra requests until deadline or cap
                while len(batch) < min(self.num_envs, self.max_eval_batch):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        nxt = self.request_q.get(timeout=remaining)
                    except _queue.Empty:
                        break
                    if isinstance(nxt, Shutdown) or getattr(nxt, "kind", None) == "shutdown":
                        # Re-queue and break — handle on next outer loop.
                        try:
                            self.request_q.put_nowait(nxt)
                        except Exception:
                            pass
                        break
                    if isinstance(nxt, PredictRequest):
                        batch.append(nxt)

                # 3) run inference, possibly in sub-chunks for safety
                self._process_batch(batch)
        except Exception:
            log.exception("[PolicyServer] crashed")
        finally:
            log.info("[PolicyServer] thread exiting")

    def _process_batch(self, batch: List[PredictRequest]):
        try:
            cap = self.max_eval_batch
            for i in range(0, len(batch), cap):
                self._forward_once(batch[i:i + cap])
        except Exception as e:
            self.zero_action_replies += len(batch)
            if self.zero_action_first_error is None:
                self.zero_action_first_error = f"{type(e).__name__}: {e}"
            log.exception("[PolicyServer] forward failed; replying with zero actions to keep "
                          "workers alive (%d zero-action replies so far)",
                          self.zero_action_replies)
            self._reply_zeros(batch)

    def _forward_once(self, items: List[PredictRequest]):
        B = len(items)
        t_start = time.monotonic()
        # ----- merge obs dicts -----
        obs_batched, batched_graph_per_mod = self._merge_obs([it.obs for it in items])

        # ----- language goals: encode any new ones via CLIP -----
        lang_emb_list = []
        for it in items:
            key = (it.env_id, it.episode_id)
            cached = self._lang_cache.get(key)
            if cached is not None:
                lang_emb_list.append(cached)
                continue
            g = it.goal_lang
            if isinstance(g, torch.Tensor):
                emb = g.to(self.method.device).float()
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                self._lang_cache[key] = emb
                lang_emb_list.append(emb)
            elif isinstance(g, str):
                import clip
                tok = clip.tokenize(g).to(self.method.device)
                with torch.no_grad():
                    emb = self.method.language_encoder.encode_text(tok).to(torch.float32)
                self._lang_cache[key] = emb
                lang_emb_list.append(emb)
            else:
                raise RuntimeError(f"goal_lang must be str or Tensor, got {type(g)}")
        lang_emb = torch.cat(lang_emb_list, dim=0)  # [B, D]

        # ----- build the batch dict expected by preprocess_batch -----
        batch_dict = {
            'observation': obs_batched,
            'goal': {'lang': lang_emb},
            'task_name': [it.task_name for it in items],
        }

        with torch.no_grad():
            state, _, goal = self.method.preprocess_batch(batch_dict)
            if self.deterministic_noise:
                noise = eval_noise(self.method, self.noise_seed,
                                   [(it.task_name, it.rollout_idx, it.step) for it in items])
                pred_action = self.method.predict(state, goal, noise=noise)
            else:
                pred_action = self.method.predict(state, goal)  # [B, act_window, action_dim]

            if self.y_bounds is not None:
                pred_action = self.method.clip_action(pred_action, self.y_bounds)

        self.node_scores.update_from_method(self.method)

        # ----- per-item sparsification slice -----
        spars_per_item = self._slice_last_sparsification(B, batched_graph_per_mod)

        # ----- reply -----
        actions_np = pred_action.detach().to('cpu').numpy().astype(np.float32)
        for i, it in enumerate(items):
            resp = ActionResponse(
                env_id=it.env_id,
                episode_id=it.episode_id,
                actions=actions_np[i],
                last_sparsification=spars_per_item[i],
            )
            self._put_reply(it.env_id, resp)

        # ----- throughput stats -----
        self._stats_batches += 1
        self._stats_items += B
        self._stats_fwd_time_s += (time.monotonic() - t_start)
        if self._stats_batches % self._stats_emit_every == 0:
            wall = time.monotonic() - self._stats_last_emit
            avg_b = self._stats_items / max(1, self._stats_batches)
            avg_ms = 1000.0 * self._stats_fwd_time_s / max(1, self._stats_batches)
            rate = self._stats_batches / max(1e-6, wall)
            print(f"[server] {self._stats_batches} batches | avg_size={avg_b:.2f} | "
                  f"avg_fwd={avg_ms:.1f}ms | {rate:.1f} batch/s", flush=True)
            self._stats_batches = 0
            self._stats_items = 0
            self._stats_fwd_time_s = 0.0
            self._stats_last_emit = time.monotonic()

    # ----- helpers --------------------------------------------------------

    def _merge_obs(self, obs_list: List[dict]):
        """Concatenate per-key obs tensors across the batch dimension.

        For PyG `Data` graph objects, use `Batch.from_data_list` and return the
        per-modality Batch so we can later slice `last_sparsification` per item.
        """
        merged = {}
        batched_graph_per_mod = {}
        keys = obs_list[0].keys()
        for k in keys:
            v0 = obs_list[0][k]
            if isinstance(v0, torch.Tensor):
                merged[k] = torch.cat([o[k] for o in obs_list], dim=0)
            elif isinstance(v0, dict):
                sub = {}
                for mod in v0.keys():
                    parts = [o[k][mod] for o in obs_list]
                    # A PyG Data must become a Batch even at B==1, so downstream
                    # graph code always sees `.batch`/`.ptr`.
                    if isinstance(parts[0], Data):
                        batch_obj = Batch.from_data_list(parts)
                        sub[mod] = batch_obj
                        batched_graph_per_mod[mod] = batch_obj
                    elif isinstance(parts[0], torch.Tensor):
                        sub[mod] = torch.cat(parts, dim=0)
                    else:
                        raise RuntimeError(
                            f"Cannot batch obs[{k}][{mod}] of type {type(parts[0])}"
                        )
                merged[k] = sub
            else:
                raise RuntimeError(f"Cannot batch obs[{k}] of type {type(v0)}")
        return merged, batched_graph_per_mod

    def _slice_last_sparsification(self, B: int, batched_graph_per_mod: dict) -> List[dict]:
        """Split `method.graph_encoder.last_sparsification` into per-batch-item dicts.

        Returns a list of length B; each entry maps modality -> {'node_names',
        'sampled_nodes'}, with `sampled_nodes` re-indexed to be local to that
        item's graph. Returns empty dicts if the encoder has no sparsification
        info (e.g. plain MultiGNN or no graph modalities).
        """
        out: List[dict] = [dict() for _ in range(B)]
        ge = getattr(self.method, 'graph_encoder', None)
        if ge is None:
            return out
        last = getattr(ge, 'last_sparsification', None)
        if not last:
            return out

        for mod, info in last.items():
            node_names = info.get('node_names')
            if not node_names:
                continue
            sampled = info.get('sampled_nodes')
            if isinstance(sampled, torch.Tensor):
                sampled = sampled.detach().cpu().numpy()
            else:
                sampled = np.asarray(sampled)

            batch_obj = batched_graph_per_mod.get(mod)
            ptr = getattr(batch_obj, 'ptr', None) if batch_obj is not None else None
            if ptr is None:
                # No batching info — can only be a single graph, so hand the
                # (flat) names and indices straight through.
                for i in range(B):
                    out[i][mod] = {
                        'node_names': list(node_names),
                        'sampled_nodes': sampled,
                    }
                continue

            # `Batch.from_data_list` collates the python-list `node_names`
            # attribute into a list-of-lists (one entry per graph), while
            # `sampled_nodes` holds *global* node indices into the batched
            # graph. Use `ptr` to map the indices back into each item's own
            # node numbering.
            ptr = ptr.detach().cpu().tolist()
            for i in range(B):
                start, end = ptr[i], ptr[i + 1]
                names_i = node_names[i] if isinstance(node_names[i], (list, tuple)) else node_names[start:end]
                mask = (sampled >= start) & (sampled < end)
                out[i][mod] = {
                    'node_names': list(names_i),
                    'sampled_nodes': sampled[mask] - start,
                }
        return out

    # Bounded put. reply_qs are maxsize=4 and this used to block, so a single
    # worker that stopped draining its queue — because it died, or abandoned a
    # rollout on a non-action response without consuming the outstanding reply —
    # stalled this thread, which then stopped draining request_q, which blocked
    # *every* worker. One stuck consumer took down the whole evaluation with
    # nothing raised anywhere. Dropping a reply instead is safe: the worker that
    # owns it is already not reading, and any worker still waiting fails loudly
    # on its own REPLY_TIMEOUT_S rather than hanging forever.
    def _put_reply(self, env_id: int, resp) -> None:
        try:
            self.reply_qs[env_id].put(resp, timeout=self.reply_put_timeout_s)
        except _queue.Full:
            self._dropped_replies += 1
            log.error(
                "[PolicyServer] reply queue for worker %d still full after %.0fs; "
                "dropping reply (%d dropped so far). That worker is not consuming — "
                "it has probably died or abandoned a rollout mid-step.",
                env_id, self.reply_put_timeout_s, self._dropped_replies,
            )

    def _reply_zeros(self, items: List[PredictRequest]):
        for it in items:
            try:
                act_dim = int(self.y_bounds.shape[1]) if self.y_bounds is not None else 12
            except Exception:
                act_dim = 12
            zeros = np.zeros((self.method.act_window_size, act_dim), dtype=np.float32)
            self._put_reply(
                it.env_id,
                ActionResponse(env_id=it.env_id, episode_id=it.episode_id, actions=zeros),
            )
