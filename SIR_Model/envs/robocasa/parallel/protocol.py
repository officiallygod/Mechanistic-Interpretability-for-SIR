"""Message types passed between env workers and the policy server."""
from dataclasses import dataclass, field
from typing import Any


# Worker -> server
KIND_PREDICT = "predict"
KIND_DONE = "done"        # worker is done, draining
KIND_ERROR = "error"      # worker hit an exception
# Server -> worker
KIND_ACTION = "action"
KIND_SHUTDOWN = "shutdown"


@dataclass
class PredictRequest:
    """One inference request from a worker.

    `obs` mirrors the batch dict consumed by `method.preprocess_batch`, but at
    batch-size 1 with CPU tensors only (GPU is server-side).
    """
    kind: str = KIND_PREDICT
    env_id: int = 0
    episode_id: int = 0
    step: int = 0
    obs: dict = field(default_factory=dict)        # {'obs_img': {...}, 'obs_graph': {...}, ...}
    goal_lang: Any = None                          # str (first call) or torch.Tensor [1, D]
    task_name: str = ""
    # Which scene this request belongs to. With `step` and `task_name` it identifies the
    # prediction independently of the worker and of `episode_id` (which counts per
    # worker), so the server can key deterministic eval noise on it.
    rollout_idx: int = 0


@dataclass
class ActionResponse:
    """Server -> one worker, in response to a PredictRequest."""
    kind: str = KIND_ACTION
    env_id: int = 0
    episode_id: int = 0
    actions: Any = None                            # np.ndarray [act_window, action_dim]
    # Per-modality sparsification slice for this batch item (or empty dict).
    # {mod_key: {'node_names': [...], 'sampled_nodes': np.ndarray[int]}}
    last_sparsification: dict = field(default_factory=dict)


@dataclass
class WorkerDone:
    kind: str = KIND_DONE
    env_id: int = 0
    # Per-task success counts produced by this worker: {task_name: n_success}.
    # `eval_n_times` is *not* divided here — the orchestrator does the averaging
    # exactly like manager.test_method does today.
    success_counts: dict = field(default_factory=dict)
    rollouts_done: dict = field(default_factory=dict)  # {task_name: n_rollouts}


@dataclass
class WorkerTaskDone:
    """Emitted after each task finishes (one per worker per task). Lets the
    orchestrator aggregate progress as it happens, so a crash or timeout
    mid-run preserves the work that did complete."""
    kind: str = "task_done"
    env_id: int = 0
    task_name: str = ""
    n_success: int = 0
    n_rollouts: int = 0
    # Per-scene success for this (worker, task): {rollout_idx: success(0/1)}.
    # Each worker owns a disjoint set of rollout_idxs for a task, so the
    # orchestrator merges these across workers with no key collisions. Enables
    # paired (per-scene) comparison across configs.
    per_scene: dict = field(default_factory=dict)
    # Per-scene sub-graph faithfulness for this (worker, task):
    # {rollout_idx: {mod_key: {'precision','recall','f1','jaccard'}}}, each value
    # already averaged over the rollout's inference steps. Empty for encoders
    # without sparsification.
    per_scene_faithfulness: dict = field(default_factory=dict)


@dataclass
class WorkerError:
    kind: str = KIND_ERROR
    env_id: int = 0
    message: str = ""


@dataclass
class Shutdown:
    kind: str = KIND_SHUTDOWN
