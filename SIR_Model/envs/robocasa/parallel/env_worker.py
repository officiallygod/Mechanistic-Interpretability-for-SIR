"""Env-worker subprocess entry point.

Each worker owns ONE MuJoCo env at a time (re-created per task) and drives a
rollout loop that ships observations to the policy server via a shared queue
instead of calling `method.predict` directly. The worker writes its own video
MP4s and sparsification PNGs, mirroring the artifact layout of the sequential
tester so downstream analysis scripts keep working.
"""
import faulthandler
import logging
import os
import queue as _queue
import traceback
from typing import List, Tuple

import cv2
import imageio
import numpy as np
import torch
import networkx as nx

from envs.robocasa.kitchen import RoboCasaKitchenTester, EmptyObservationError
from envs.robocasa.utils import (
    accumulate_faithfulness, mean_faithfulness, per_rollout_seed,
    rollout_artifact_dir, seed_rollout, visualize_sparsification_rollout,
)
from envs.robocasa.parallel.protocol import (
    PredictRequest,
    WorkerDone,
    WorkerError,
    WorkerTaskDone,
)

log = logging.getLogger(__name__)

# How many blank render buffers IN A ROW mean the EGL context is dead rather than the
# policy being blind.
#
# A bad policy can blank several rollouts in a row on its own: one that selects a tiny
# subgraph blanked its first three rollouts on every worker, and a threshold of 3 then
# killed every worker and reported 0/200 for the whole unit. Only a dead device blanks
# eight in a row.
#
# The real discriminator is not the run length: a dead EGL context usually SIGABRTs
# inside mjr_readPixels and the worker dies with exit code -6 whether or not we act.
# This counter is a backstop for the half-dead state where the context returns blank
# buffers without aborting, so it is set well clear of anything a policy can do.
MAX_CONSECUTIVE_BLANKS = 8

# Upper bound on how long a worker waits for the server's reply to one request.
# A forward pass is milliseconds, so this only ever fires when the request never
# reached the server at all — `mp.Queue.put` pickles on a background feeder
# thread, so an unpicklable observation is dropped there with only a traceback on
# stderr, and the worker would otherwise block here forever.
REPLY_TIMEOUT_S = 300.0

# The request queue is bounded (2 * num_envs), so `put` blocks once it fills. It
# had no timeout, and that is what turns one stuck consumer into a silent
# whole-run hang: if the server stops draining request_q, every worker blocks
# *here*, never reaches `reply_q.get`, and so never trips REPLY_TIMEOUT_S. The
# processes stay alive and idle with nothing raised. Generous, because a full
# queue is normal back-pressure while the server runs a batch; this only has to
# catch a server that is never coming back.
REQUEST_TIMEOUT_S = 300.0


def _maybe_load_graph_encoders(graph_modalities, pretrained_img_encoder_name):
    """Lazily build (text_encoder, cropped_image_feature_encoder) only if needed.

    These are big and we don't want to pay for them when the rollout config
    doesn't use graph modalities.
    """
    text_encoder = (None, None)
    cropped_encoder = None
    if not graph_modalities:
        return text_encoder, cropped_encoder
    if "siglip_text_emb" in graph_modalities:
        from transformers import SiglipModel, SiglipProcessor
        model_id = "google/siglip-base-patch16-224"
        text_encoder = (
            SiglipModel.from_pretrained(model_id),
            SiglipProcessor.from_pretrained(model_id),
        )
    if "cropped_image_feature" in graph_modalities:
        from networks.vision_encoder.utils import load_pretrained_image_encoder
        cropped_encoder = load_pretrained_image_encoder(pretrained_img_encoder_name)
    return text_encoder, cropped_encoder


def _build_obs(tester, manager_cfg, obs_all, left_graph, right_graph):
    """Construct the per-step observation dict, mirroring kitchen.py:76-84."""
    obs = {}
    if manager_cfg.use_proprioceptive:
        obs['obs_prop'] = tester.create_prop_obs(obs_all, manager_cfg)
    if manager_cfg.use_image:
        obs['obs_img'] = tester.create_image_obs(obs_all, manager_cfg)
    if manager_cfg.use_graph:
        obs['obs_graph'] = tester.create_graph_obs(obs_all, left_graph, right_graph, manager_cfg)
    return obs


# Seeding lives in envs.robocasa.utils so the sequential tester shares it.
_per_rollout_seed = per_rollout_seed


def _run_one_rollout(tester, manager_cfg, env, task_name, env_id, episode_id,
                     request_q, reply_q, store_videos, working_dir, test_count,
                     epoch, rollout_idx, base_seed):
    """Execute one full rollout (reset -> success or horizon) on `env`.

    Returns (success flag 0/1, faithfulness dict {mod: {metric: mean}}).
    """
    # Force a deterministic env RNG state for this (task, rollout_idx) so the
    # resulting scene is reproducible across rounds and runs.
    seed = _per_rollout_seed(task_name, rollout_idx, base_seed)
    seed_rollout(env, seed)
    obs_all = env.reset()

    # cls_list / cls_numbers are computed per-reset (matches kitchen.py:62-66)
    tester.cls_list = {cls: i for i, cls in enumerate(list(env.model.classes_to_ids.keys()))}
    cls_numbers = []
    for class_name in list(tester.cls_list.keys()):
        cls_numbers.append(tester.cls_list[class_name] + 1)
    tester.cls_numbers = torch.tensor(cls_numbers)

    # Per-modality NetworkX accumulators (per episode).
    left_graph, right_graph = {}, {}
    for mod in manager_cfg.graph_modalities:
        left_graph[mod] = nx.DiGraph()
        right_graph[mod] = nx.DiGraph()

    lang_goal = env.get_ep_meta()['lang']

    obs = _build_obs(tester, manager_cfg, obs_all, left_graph, right_graph)

    presence_counts = {}
    selection_timeline = {}
    total_rollout_steps = 0
    faith_sums = {}
    faith_steps = {}
    frames = []

    success = 0
    step_counter = 0
    # The server caches the lang embedding by (env_id, episode_id); we only
    # need to ship the string on the first request of this episode.
    goal_lang_to_send = lang_goal

    while step_counter < env.horizon:
        # ----- send obs to server, await action chunk -----
        req = PredictRequest(
            env_id=env_id,
            episode_id=episode_id,
            step=step_counter,
            obs=obs,
            goal_lang=goal_lang_to_send,
            task_name=task_name,
            rollout_idx=rollout_idx,
        )
        try:
            request_q.put(req, timeout=REQUEST_TIMEOUT_S)
        except _queue.Full:
            raise RuntimeError(
                f"policy server not draining requests within "
                f"{REQUEST_TIMEOUT_S:.0f}s (task={task_name}, "
                f"rollout={rollout_idx}, step={step_counter}); the server is "
                f"most likely blocked writing to another worker's reply queue"
            )
        try:
            resp = reply_q.get(timeout=REPLY_TIMEOUT_S)
        except _queue.Empty:
            raise RuntimeError(
                f"no reply from policy server within {REPLY_TIMEOUT_S:.0f}s "
                f"(task={task_name}, rollout={rollout_idx}, step={step_counter}); "
                f"the request was most likely dropped while being pickled — check "
                f"stderr for a traceback from multiprocessing/queues.py"
            )
        if getattr(resp, 'kind', None) != 'action':
            # Shutdown or unexpected message — abort rollout.
            return success, mean_faithfulness(faith_sums, faith_steps)
        # subsequent steps in same episode: no need to resend the goal text
        goal_lang_to_send = None

        # ----- sparsification accumulation (post-forward) -----
        if manager_cfg.use_graph and resp.last_sparsification:
            for mod_key, info in resp.last_sparsification.items():
                node_names = info.get('node_names')
                if not node_names:
                    continue
                bucket = presence_counts.setdefault(mod_key, {})
                step_names = [node_names[int(idx)] for idx in info['sampled_nodes']
                              if int(idx) < len(node_names)]
                for name in step_names:
                    bucket[name] = bucket.get(name, 0) + 1
                selection_timeline.setdefault(mod_key, []).append(step_names)
                accumulate_faithfulness(
                    faith_sums, faith_steps, mod_key, step_names,
                    node_names, task_name, lang_goal,
                )
            total_rollout_steps += 1

        # Actions arrive already clipped by the server (method.clip_action).
        pred_action = resp.actions  # np [act_window, action_dim]
        for a in range(pred_action.shape[0]):
            if store_videos:
                img = obs_all['robot0_agentview_left_image']
                frames.append(cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC))
            obs_all, _, _, _ = env.step(pred_action[a].astype(np.float32).reshape(-1))
            step_counter += 1

        # rebuild obs for next iteration
        obs = _build_obs(tester, manager_cfg, obs_all, left_graph, right_graph)

        if env._check_success():
            success = 1
            break

    # ----- write per-rollout artifacts -----
    add_str = f"epoch_{epoch}_" if epoch is not None else ''
    if store_videos and frames:
        video_filename = add_str + f"env_{task_name}_{test_count}_rollout_{rollout_idx}_w{env_id}_success_{success}.mp4"
        try:
            video_dir = rollout_artifact_dir(working_dir, 'videos', task_name, test_count)
            imageio.mimsave(os.path.join(video_dir, video_filename), frames, fps=30)
        except Exception:
            log.exception(f"[worker {env_id}] failed to save video")

    if (presence_counts and total_rollout_steps > 0
            and getattr(manager_cfg, 'store_sparsification_plots', False)):
        spars_dir = rollout_artifact_dir(working_dir, 'sparsification', task_name, test_count)
        for mod_key, counts in presence_counts.items():
            png_name = add_str + f"env_{task_name}_{test_count}_rollout_{rollout_idx}_{mod_key}_w{env_id}_success_{success}.png"
            try:
                visualize_sparsification_rollout(
                    counts,
                    total_rollout_steps,
                    os.path.join(spars_dir, png_name),
                    timeline=selection_timeline.get(mod_key),
                    title=f"{task_name} rollout {rollout_idx} — {mod_key}",
                )
            except Exception:
                log.exception(f"[worker {env_id}] failed to save sparsification png")

    return success, mean_faithfulness(faith_sums, faith_steps)


def env_worker_main(env_id: int,
                    assignments: List[Tuple[str, List[int]]],
                    manager_cfg,
                    request_q,
                    reply_q,
                    done_q,
                    working_dir: str,
                    base_seed: int,
                    store_videos: bool,
                    test_count: int,
                    epoch):
    """Process entry point. `assignments` is a list of (task_name, [rollout_idx,...])."""
    try:
        # Spawned children bypass the parent's log capture, so a crash here —
        # including a segfault, via faulthandler — would otherwise vanish into the
        # terminal instead of landing next to the run.
        err_dir = os.path.join(working_dir, 'worker_logs')
        os.makedirs(err_dir, exist_ok=True)
        err_f = open(os.path.join(err_dir, f'worker{env_id}_round{test_count}.err'),
                     'a', buffering=1)
        os.dup2(err_f.fileno(), 2)
        faulthandler.enable()

        # Workers stay CPU-only — never touch CUDA, even by accident.
        torch.set_num_threads(1)  # avoid OMP oversubscription against the server
        # Workers only run inference. An obs tensor that still carries autograd
        # history cannot be pickled onto the request queue (the cropped-image
        # feature encoder produces one), and `mp.Queue.put` drops it on its feeder
        # thread with nothing raised here — the worker then simply waits out
        # REPLY_TIMEOUT_S.
        torch.set_grad_enabled(False)

        text_encoder, cropped_encoder = _maybe_load_graph_encoders(
            manager_cfg.graph_modalities, manager_cfg.pretrained_img_encoder_name,
        )

        success_counts = {t: 0 for t, _ in assignments}
        rollouts_done = {t: 0 for t, _ in assignments}
        # Per-scene success: {task: {rollout_idx: success(0/1)}}, for paired eval.
        per_scene = {t: {} for t, _ in assignments}
        # Per-scene sub-graph faithfulness: {task: {rollout_idx: {mod: {metric: v}}}}.
        per_scene_faith = {t: {} for t, _ in assignments}
        episode_counter = 0

        for task_name, rollout_idxs in assignments:
            if not rollout_idxs:
                continue
            # Seed per-worker so different workers construct different env init
            # RNGs; the per-rollout seed below is what actually pins the scene.
            seed = (base_seed + env_id * 100003 + (hash(task_name) & 0xFFFF)) & 0x7FFFFFFF
            try:
                tester = RoboCasaKitchenTester(
                    dataset_path=manager_cfg.data_path,
                    task_list=[task_name],
                    use_depth=False,
                    cropped_image_feature_encoder=cropped_encoder,
                    text_encoder=text_encoder,
                    seed=seed,
                )
                env = tester.env_list[0]
            except Exception as e:
                log.exception(f"[worker {env_id}] failed to construct env for {task_name}")
                done_q.put(WorkerError(
                    env_id=env_id,
                    message=f"env construct for {task_name}: {e}\n{traceback.format_exc()}",
                ))
                continue

            n_task = len(rollout_idxs)
            # Consecutive blank-buffer count. Reset by any rollout that completes, so it
            # measures a RUN of blanks (dead device) rather than a total (which a bad
            # policy can also run up over a long eval).
            consecutive_blanks = 0
            blank_run = []          # rollout_idxs of the current run of blanks
            print(f"[w={env_id}] starting {task_name}: {n_task} rollouts", flush=True)
            for i_local, rollout_idx in enumerate(rollout_idxs):
                episode_counter += 1
                episode_id = env_id * 1_000_000 + episode_counter
                try:
                    success, rollout_faith = _run_one_rollout(
                        tester=tester,
                        manager_cfg=manager_cfg,
                        env=env,
                        task_name=task_name,
                        env_id=env_id,
                        episode_id=episode_id,
                        request_q=request_q,
                        reply_q=reply_q,
                        store_videos=store_videos,
                        working_dir=working_dir,
                        test_count=test_count,
                        epoch=epoch,
                        rollout_idx=rollout_idx,
                        base_seed=base_seed,
                    )
                    success_counts[task_name] += int(success)
                    tag = "OK " if success else "fail"
                    consecutive_blanks = 0
                    blank_run = []
                except EmptyObservationError as e:
                    # A blank segmentation buffer has TWO causes needing opposite handling.
                    #
                    #  DEAD RENDERER -- the EGL context is gone, and every later rollout
                    #    on this worker blanks too. Scoring those would score a hardware
                    #    fault as a policy failure. The worker must DIE so the retry round
                    #    respawns one with a fresh GL context.
                    #
                    #  POLICY DROVE THE CAMERA BLIND -- a flailing policy moves the arm to
                    #    a pose where no known object is in view. Isolated, scene- and
                    #    policy-specific. That is a policy failure and belongs in the
                    #    average as a zero.
                    #
                    # Separable by persistence: a dead context never recovers, so score
                    # isolated blanks as failures and raise once they run together.
                    consecutive_blanks += 1
                    if consecutive_blanks >= MAX_CONSECUTIVE_BLANKS:
                        # The earlier blanks of this run were almost certainly the same
                        # dead device, not the policy. Un-score them so the retry round
                        # re-runs those scenes on a fresh worker. Scenes are seeded per
                        # rollout index, so a genuinely policy-blind scene blanks again
                        # and still scores 0.
                        for ri in blank_run:
                            if per_scene[task_name].pop(ri, None) is not None:
                                rollouts_done[task_name] -= 1
                            per_scene_faith[task_name].pop(ri, None)
                        log.error(
                            f"[worker {env_id}] rollout {rollout_idx} of {task_name}: "
                            f"{consecutive_blanks} CONSECUTIVE blank buffers ({e}) -- "
                            f"GL context is dead, not a policy failure; exiting so the "
                            f"retry round respawns this worker.")
                        print(f"[w={env_id}] {task_name} rollout {rollout_idx} -> BLANK "
                              f"x{consecutive_blanks} (worker exiting for respawn)",
                              flush=True)
                        # Flush what this worker already scored BEFORE dying. Results are
                        # otherwise sent once, after the task's whole rollout loop, so a
                        # worker exiting mid-task would discard every rollout it completed.
                        # The collector merges per_scene by rollout index and the retry
                        # round re-runs only the indices still missing, so nothing is
                        # counted twice.
                        try:
                            done_q.put(WorkerTaskDone(
                                env_id=env_id,
                                task_name=task_name,
                                n_success=success_counts.get(task_name, 0),
                                n_rollouts=rollouts_done.get(task_name, 0),
                                per_scene=per_scene.get(task_name, {}),
                                per_scene_faithfulness=per_scene_faith.get(task_name, {}),
                            ))
                            log.warning(
                                f"[worker {env_id}] flushed {len(per_scene.get(task_name, {}))} "
                                f"scored rollout(s) for {task_name} before exiting")
                        except Exception:
                            log.exception(f"[worker {env_id}] failed to flush partial "
                                          f"results for {task_name} before exiting")
                        raise
                    log.warning(
                        f"[worker {env_id}] rollout {rollout_idx} of {task_name}: blank "
                        f"buffer ({e}) -- scoring as a POLICY FAILURE (the policy drove "
                        f"the camera somewhere with no known object). Raises if it "
                        f"repeats {MAX_CONSECUTIVE_BLANKS} times in a row.")
                    success = 0
                    rollout_faith = {}
                    tag = "fail (blind)"
                    blank_run.append(rollout_idx)
                except Exception:
                    log.exception(f"[worker {env_id}] rollout {rollout_idx} of {task_name} crashed")
                    # Continue with remaining rollouts; treat this one as a failure.
                    success = 0
                    rollout_faith = {}
                    tag = "CRASH"
                rollouts_done[task_name] += 1
                per_scene[task_name][rollout_idx] = int(success)
                if rollout_faith:
                    per_scene_faith[task_name][rollout_idx] = rollout_faith
                print(f"[w={env_id}] {task_name} rollout {rollout_idx} "
                      f"({i_local + 1}/{n_task}) -> {tag} | task acc so far: "
                      f"{success_counts[task_name]}/{rollouts_done[task_name]}", flush=True)

            # Report this task's totals so the orchestrator can aggregate
            # progress even if the worker is later killed or hangs.
            try:
                done_q.put(WorkerTaskDone(
                    env_id=env_id,
                    task_name=task_name,
                    n_success=success_counts.get(task_name, 0),
                    n_rollouts=rollouts_done.get(task_name, 0),
                    per_scene=per_scene.get(task_name, {}),
                    per_scene_faithfulness=per_scene_faith.get(task_name, {}),
                ))
            except Exception:
                log.exception(f"[worker {env_id}] failed to put WorkerTaskDone for {task_name}")

            try:
                env.close()
            except Exception:
                pass

        done_q.put(WorkerDone(env_id=env_id, success_counts=success_counts, rollouts_done=rollouts_done))
    except Exception as e:
        try:
            done_q.put(WorkerError(env_id=env_id, message=f"worker fatal: {e}\n{traceback.format_exc()}"))
        except Exception:
            pass
        # Always emit WorkerDone so the orchestrator's `remaining` counter
        # decrements — otherwise it hangs waiting for this worker forever.
        try:
            done_q.put(WorkerDone(env_id=env_id, success_counts={}, rollouts_done={}))
        except Exception:
            pass
