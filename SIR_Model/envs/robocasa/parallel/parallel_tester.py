"""Parallel evaluation orchestrator.

Sibling to `RoboCasaKitchenTester`. Same return shape:
    {f"Avg_Test_Reward_{task}": success_rate, ...}

Activated when `parallel_eval=True` and `num_envs > 1`. With `num_envs == 1`
falls back through the same machinery but with a single worker — useful for
verifying the parallel path is numerically close to the sequential baseline.
"""
import logging
import queue as _queue
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.multiprocessing as mp

from envs.robocasa.parallel.env_worker import env_worker_main
from envs.robocasa.parallel.policy_server import PolicyServer
from envs.robocasa.parallel.protocol import (
    Shutdown,
    WorkerDone,
    WorkerError,
    WorkerTaskDone,
)
from envs.robocasa.utils import TASK_LIST

log = logging.getLogger(__name__)


@dataclass
class ManagerConfig:
    """Picklable subset of `RoboCasa_Manager` that workers need to build obs.

    Mirrors attribute names so the same obs-builders (`kitchen.py`) can consume
    either object.
    """
    data_path: str
    use_proprioceptive: bool
    use_image: bool
    use_graph: bool
    prop_modalities: list
    image_modalities: list
    graph_modalities: list
    use_graph_fusion: bool
    use_splitted_modalities: bool
    is_cropped_fusion: bool
    pretrained_img_encoder_name: str
    y_bounds: torch.Tensor
    store_sparsification_plots: bool = False


def _build_manager_cfg(manager) -> ManagerConfig:
    return ManagerConfig(
        data_path=manager.data_path,
        use_proprioceptive=manager.use_proprioceptive,
        use_image=manager.use_image,
        use_graph=manager.use_graph,
        prop_modalities=list(manager.prop_modalities or []),
        image_modalities=list(manager.image_modalities or []),
        graph_modalities=list(manager.graph_modalities or []),
        use_graph_fusion=getattr(manager, 'use_graph_fusion', False),
        use_splitted_modalities=getattr(manager, 'use_splitted_modalities', False),
        is_cropped_fusion=getattr(manager, 'is_cropped_fusion', False),
        pretrained_img_encoder_name=getattr(manager, 'pretrained_img_encoder_name', '') or '',
        y_bounds=manager.y_bounds.clone(),
        store_sparsification_plots=getattr(manager, 'store_sparsification_plots', False),
    )


def _missing_rollouts(per_task_scene, task_names, eval_n_times):
    """{task: [rollout_idx, ...]} for every requested rollout no worker has reported."""
    missing = {}
    for t in task_names:
        have = per_task_scene.get(t, {})
        want = [i for i in range(eval_n_times) if i not in have]
        if want:
            missing[t] = want
    return missing


def _build_retry_assignments(missing, num_envs):
    """Spread the missing (task, rollout_idx) pairs round-robin over at most num_envs
    workers, in the same [(task, [rollout_idx, ...]), ...] shape as `_build_assignments`.
    Only as many workers as there are missing rollouts: each costs a full env startup."""
    flat = [(t, i) for t, idxs in missing.items() for i in idxs]
    n = max(1, min(num_envs, len(flat)))
    retry = [[] for _ in range(n)]
    for j, (t, i) in enumerate(flat):
        w = j % n
        for task, idxs in retry[w]:
            if task == t:
                idxs.append(i)
                break
        else:
            retry[w].append((t, [i]))
    return retry


def _build_assignments(task_names: List[str], eval_n_times: int, num_envs: int
                       ) -> List[List[Tuple[str, List[int]]]]:
    """Round-robin the `eval_n_times` rollouts per task across `num_envs` workers.

    Returns a list of length num_envs; entry w is a list of (task, [rollouts]).
    """
    out: List[List[Tuple[str, List[int]]]] = [[] for _ in range(num_envs)]
    for task in task_names:
        worker_idxs = [[] for _ in range(num_envs)]
        for j in range(eval_n_times):
            worker_idxs[j % num_envs].append(j)
        for w in range(num_envs):
            if worker_idxs[w]:
                out[w].append((task, worker_idxs[w]))
    return out


class RoboCasaKitchenParallelTester:
    """Spawns N MuJoCo env workers + a policy-server thread, returns the same
    `{Avg_Test_Reward_<task>: rate}` dict shape as `RoboCasaKitchenTester.test`.
    """

    def __init__(self,
                 num_envs: int,
                 batch_timeout_ms: float = 20.0,
                 max_eval_batch: int = 16,
                 worker_seed_base: int = 42,
                 deterministic_noise: bool = False,
                 noise_seed: int = None,
                 ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self.batch_timeout_ms = batch_timeout_ms
        self.max_eval_batch = max_eval_batch
        self.worker_seed_base = worker_seed_base
        self.deterministic_noise = deterministic_noise
        # Defaults to the scene seed; separate so noise sensitivity can be measured on
        # identical scenes by varying this alone.
        self.noise_seed = worker_seed_base if noise_seed is None else noise_seed
        self.per_scene_success: Dict[str, Dict[int, int]] = {}
        self.per_scene_faithfulness: Dict[str, Dict[int, dict]] = {}

    def _run_round(self, assignments_per_worker, method, manager, store_videos,
                   working_dir, test_count, epoch,
                   per_task_success, per_task_count, per_task_scene, per_task_faith,
                   errors):
        """One spawn/drain/shutdown cycle, merging results into the caller's dicts.

        Extracted so a retry round for rollouts lost to a dead worker runs exactly the
        same path as the first round. Returns the round's PolicyServer, whose
        zero-action count and node scores the caller accumulates across rounds.
        """
        # Sized by THIS round's workers, not self.num_envs: a retry round may need
        # only a few, and the server batches up to that many requests at a time.
        n_workers = len(assignments_per_worker)

        # Spawn context — fork breaks CUDA + MuJoCo.
        ctx = mp.get_context('spawn')
        request_q = ctx.Queue(maxsize=max(2 * n_workers, 4))
        done_q = ctx.Queue(maxsize=max(n_workers + 4, 8))
        reply_qs = [ctx.Queue(maxsize=4) for _ in range(n_workers)]

        manager_cfg = _build_manager_cfg(manager)

        # Start workers BEFORE the server thread to ensure the queues are
        # established before the first put.
        workers = []
        for env_id, assignments in enumerate(assignments_per_worker):
            p = ctx.Process(
                target=env_worker_main,
                args=(
                    env_id,
                    assignments,
                    manager_cfg,
                    request_q,
                    reply_qs[env_id],
                    done_q,
                    working_dir,
                    self.worker_seed_base,
                    store_videos,
                    test_count,
                    epoch,
                ),
                daemon=False,
            )
            p.start()
            workers.append(p)

        server = PolicyServer(
            method=method,
            request_q=request_q,
            reply_qs=reply_qs,
            num_envs=n_workers,
            batch_timeout_ms=self.batch_timeout_ms,
            max_eval_batch=self.max_eval_batch,
            y_bounds=manager.y_bounds.clone(),
            deterministic_noise=self.deterministic_noise,
            noise_seed=self.noise_seed,
        )
        server.start()

        # Drain done_q until every worker has emitted WorkerDone. Aggregate
        # per-task success counts as `WorkerTaskDone` messages arrive so that
        # partial work is preserved even if a worker later crashes / hangs.
        remaining = len(workers)

        # No global wall-clock cap — long evals can take many hours. Instead
        # poll done_q with a short timeout and use process liveness as the
        # abort signal: if every worker has exited yet `remaining > 0`, the
        # workers crashed without emitting WorkerDone; stop waiting then.
        poll_s = 30.0
        last_msg_time = time.monotonic()
        # Stall detector: if we go this long with no message *and* all workers
        # appear stuck (alive but silent), we abort. This shouldn't fire in
        # normal use because each rollout produces a WorkerTaskDone within
        # minutes; tune up if your env steps are slower.
        stall_s = 60.0 * 60.0 * 4.0  # 4h
        while remaining > 0:
            try:
                msg = done_q.get(timeout=poll_s)
                last_msg_time = time.monotonic()
            except _queue.Empty:
                # Liveness check.
                alive = [p for p in workers if p.is_alive()]
                if not alive:
                    codes = {f"w{i}": p.exitcode for i, p in enumerate(workers)}
                    log.error(f"[ParallelTester] all workers exited but {remaining} WorkerDone(s) "
                              f"still missing — treating as crash, aborting wait. "
                              f"Exit codes: {codes} (1 = python error before the worker entry "
                              f"point, e.g. import failure; negative = killed by signal). "
                              f"Workers re-import the source tree at every round — do not "
                              f"switch branches or edit worker code while a run is active. "
                              f"Per-worker stderr is in <working_dir>/worker_logs/.")
                    break
                if (time.monotonic() - last_msg_time) > stall_s:
                    log.error(f"[ParallelTester] no progress for {stall_s/60:.0f} min "
                              f"with {len(alive)} workers still alive — aborting wait.")
                    break
                continue

            if isinstance(msg, WorkerError):
                errors.append(f"worker {msg.env_id}: {msg.message}")
                log.error(f"[ParallelTester] {errors[-1]}")
                continue
            if isinstance(msg, WorkerTaskDone):
                per_task_success[msg.task_name] = per_task_success.get(msg.task_name, 0) + int(msg.n_success)
                per_task_count[msg.task_name] = per_task_count.get(msg.task_name, 0) + int(msg.n_rollouts)
                scene_map = per_task_scene.setdefault(msg.task_name, {})
                for ri, s in (msg.per_scene or {}).items():
                    scene_map[ri] = scene_map.get(ri, 0) + int(s)
                # Workers own disjoint rollout_idxs per task, so this is a plain merge.
                faith_map = per_task_faith.setdefault(msg.task_name, {})
                faith_map.update(getattr(msg, 'per_scene_faithfulness', None) or {})
                log.info(f"[ParallelTester] w={msg.env_id} done {msg.task_name}: "
                         f"{msg.n_success}/{msg.n_rollouts}  "
                         f"(task total: {per_task_success[msg.task_name]}/{per_task_count[msg.task_name]})")
                continue
            if isinstance(msg, WorkerDone):
                # Final per-worker signal. Per-task counts have already been
                # aggregated from WorkerTaskDone, so ignore msg.success_counts
                # here to avoid double-counting.
                remaining -= 1
                log.info(f"[ParallelTester] worker {msg.env_id} finished "
                         f"({len(workers) - remaining}/{len(workers)})")
                continue

        # Shut down server thread.
        server.stop()

        # Tell each worker explicitly to shut down (in case they're still hanging
        # on reply_q — shouldn't happen if WorkerDone has been emitted, but cheap).
        for q in reply_qs:
            try:
                q.put_nowait(Shutdown())
            except Exception:
                pass

        # Join workers.
        for p in workers:
            p.join(timeout=30.0)
            if p.is_alive():
                log.warning(f"[ParallelTester] worker pid={p.pid} still alive after join; terminating")
                p.terminate()
                p.join(timeout=5.0)

        return server

    def test(self, method, manager, store_videos: bool, eval_n_times: int,
             working_dir: str, test_count: int, epoch, task_names: List[str] = None):
        # Caller (`RoboCasa_Manager.test_method`) may pass an explicit task list
        # (e.g. ["CloseDrawer"] during in-training eval). Fall back to the
        # manager's configured task list otherwise.
        if task_names is None:
            task_names = list(manager.task_names) if manager.task_names else []
        if task_names and task_names[0] == "ALL":
            task_names = list(TASK_LIST)

        assignments_per_worker = _build_assignments(task_names, eval_n_times, self.num_envs)

        per_task_success: Dict[str, int] = {t: 0 for t in task_names}
        per_task_count: Dict[str, int] = {t: 0 for t in task_names}
        # Per-scene success: {task: {rollout_idx: success(0/1)}}.
        per_task_scene: Dict[str, Dict[int, int]] = {t: {} for t in task_names}
        # Per-scene faithfulness: {task: {rollout_idx: {mod: {metric: v}}}}.
        per_task_faith: Dict[str, Dict[int, dict]] = {t: {} for t in task_names}
        errors: List[str] = []

        server = self._run_round(
            assignments_per_worker, method, manager, store_videos, working_dir,
            test_count, epoch, per_task_success, per_task_count,
            per_task_scene, per_task_faith, errors)
        zero_action_replies = getattr(server, 'zero_action_replies', 0)
        zero_action_first_error = getattr(server, 'zero_action_first_error', None)
        node_scores = server.node_scores

        # ---- retry any rollouts lost to a dead worker ----
        # A MuJoCo worker can die with a C-level abort inside the EGL renderer
        # (`mju_error` in mjr_readPixels) -- SIGABRT, so no Python exception is raisable
        # and the process disappears along with every rollout it had not yet reported.
        # One death takes a whole worker's remaining assignment with it, and because the
        # rate below divides by the REQUESTED count those rollouts silently count as
        # failures. Nothing here can prevent the abort; what it can do is notice which
        # rollouts are missing and run them again with fresh workers.
        max_retry_rounds = 2
        for attempt in range(1, max_retry_rounds + 1):
            missing = _missing_rollouts(per_task_scene, task_names, eval_n_times)
            if not missing:
                break
            n_missing = sum(len(v) for v in missing.values())
            log.warning(
                "[ParallelTester] retry round %d/%d -- %d rollout(s) never reported "
                "across %d task(s), most likely workers lost to a renderer abort "
                "(see worker_logs/*.err for 'Fatal Python error'). Re-running just those.",
                attempt, max_retry_rounds, n_missing, len(missing))
            retry_server = self._run_round(
                _build_retry_assignments(missing, self.num_envs), method, manager,
                store_videos, working_dir, test_count, epoch, per_task_success,
                per_task_count, per_task_scene, per_task_faith, errors)
            zero_action_replies += getattr(retry_server, 'zero_action_replies', 0)
            if zero_action_first_error is None:
                zero_action_first_error = getattr(retry_server, 'zero_action_first_error', None)
            node_scores.merge(retry_server.node_scores)

            # A round that recovers NOTHING means the fault is not a per-worker death
            # that a fresh process fixes -- it is the GPU/EGL device itself. Respawned
            # workers inherit the poisoned device and abort at the same place, so every
            # further round would recover nothing too. Stop instead of burning the time.
            still_missing = sum(len(v) for v in
                                _missing_rollouts(per_task_scene, task_names, eval_n_times).values())
            if still_missing == n_missing:
                log.error(
                    "[ParallelTester] retry round %d recovered 0 of %d missing "
                    "rollout(s) -- this is a dead render device, not a lost worker; "
                    "further rounds cannot help. Re-run this evaluation as a fresh job, "
                    "on a different node if possible.", attempt, n_missing)
                break

        # ---- Build result dict (same shape as sequential tester) ----
        # `RoboCasaKitchenTester.test` returns success_count / eval_n_times per
        # task. We mirror that exactly: divide by the *requested* eval_n_times,
        # not by rollouts_done — that way a worker crash shows up as a lower
        # number, identical to how the sequential path treats a crashed env.
        result_dict = {}
        for task in task_names:
            total = per_task_success.get(task, 0)
            denom = max(1, eval_n_times)
            result_dict[f"Avg_Test_Reward_{task}"] = total / denom
            log.info(f"Total reward in {task}: {total} / {eval_n_times} "
                     f"(rollouts run: {per_task_count.get(task, 0)})")

        # A lost worker is not always a logged crash. When a worker dies during
        # CUDA context allocation the process goes down with no Python-level
        # exception, so `errors` stays empty and the only trace is that a fifth
        # of the rollouts never ran — which the line above prints but nothing
        # acts on. Because the rate is divided by the *requested* eval_n_times,
        # such a run comes back quietly understated rather than obviously
        # broken. Say so loudly.
        shortfall = {
            t: (per_task_count.get(t, 0), eval_n_times)
            for t in task_names
            if per_task_count.get(t, 0) < eval_n_times
        }
        if shortfall:
            log.error(
                "[ParallelTester] INCOMPLETE EVALUATION — %d/%d task(s) ran "
                "fewer rollouts than requested; every Avg_Test_Reward above is "
                "divided by the requested count and is therefore UNDERSTATED. "
                "Pair from eval_per_scene_<task>.json, which records only the "
                "rollouts that actually ran. Shortfalls: %s",
                len(shortfall), len(task_names),
                ", ".join(f"{t}: {got}/{want}" for t, (got, want) in sorted(shortfall.items())),
            )

        # Zero-action replies mean some rollouts were driven by a policy that was not
        # actually consulted. They do not crash anything and they do not shorten the
        # rollout count, so nothing above notices — the run just comes back quietly
        # worse. On 2026-08-26 a single CUDA OOM in the server did this mid-eval.
        zeros = zero_action_replies
        if zeros:
            log.error(
                "[ParallelTester] DEGRADED EVALUATION — the policy server failed %d "
                "forward pass(es) and answered with ZERO actions; the affected rollouts "
                "were stepped with a null policy and count as failures, so every "
                "Avg_Test_Reward below is a LOWER BOUND. First failure: %s",
                zeros, zero_action_first_error or 'unknown',
            )

        if errors:
            log.warning(f"[ParallelTester] {len(errors)} worker error(s) during eval")

        # Stash per-scene results for this round so the manager can accumulate
        # them across `times_repeat` rounds and write eval_per_scene_<task>.json.
        self.per_scene_success = per_task_scene
        self.node_scores = node_scores
        self.per_scene_faithfulness = per_task_faith

        return result_dict
