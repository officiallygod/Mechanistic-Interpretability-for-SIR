"""Eval robustness: blank views, dead workers, and rollouts lost mid-task.

Three failures that each produced silently wrong or missing numbers, tested without
MuJoCo -- the env, the rollout and the worker round are replaced by fakes, because the
logic under test is the bookkeeping around them, not the simulation.

  1. A view with no known object must raise a typed EmptyObservationError, not an
     untyped unpack ValueError indistinguishable from a real bug.
  2. The worker scores an isolated blank as a policy failure, but on a run of
     MAX_CONSECUTIVE_BLANKS (a dead render context) it flushes what it already scored,
     un-scores the run, and exits so a fresh worker can redo those scenes.
  3. The parallel tester re-runs rollouts no worker reported, stops when a round
     recovers nothing, and builds Avg_Test_Reward AFTER the retries.

Run: python tests/test_eval_robustness.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

import numpy as np

from envs.robocasa.kitchen import EmptyObservationError, RoboCasaKitchenTester
from envs.robocasa.parallel import env_worker, parallel_tester
from envs.robocasa.parallel.protocol import WorkerDone, WorkerError, WorkerTaskDone
from utils.node_scores import NodeScoreTracker


# ------------------------------------------------------------------ 1. blank views
class _Bare(RoboCasaKitchenTester):
    """Only the image helpers, without building a MuJoCo env."""
    def __init__(self):
        from torchvision import transforms
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((128, 128))])
        self.cropped_image_feature_encoder = None


def _obs(blank):
    """A frame whose segmentation contains no known class when `blank`."""
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    seg = (np.zeros((128, 128, 1), dtype=np.int32) if blank
           else np.full((128, 128, 1), 1, np.int32))
    return {f"robot0_agentview_{v}_image": img for v in ("left", "right")} | \
           {f"robot0_agentview_{v}_segmentation_class": seg for v in ("left", "right")}


def test_empty_view_raises_typed_error():
    t = _Bare()
    try:
        t.helper_function(_obs(blank=True), "agentview_left", [1], {"Counter": 0})
    except EmptyObservationError:
        return
    except ValueError as e:
        raise AssertionError(f"regressed to the untyped unpack failure: {e}")
    raise AssertionError("a blind view must raise EmptyObservationError")


def test_non_empty_view_still_returns_four():
    t = _Bare()
    out = t.helper_function(_obs(blank=False), "agentview_left", [1], {"Counter": 0})
    assert len(out) == 4
    assert len(out[0]) >= 1, "a populated segmentation should yield at least one object"


# ------------------------------------------------------------------ 2. the worker
class _Q:
    def __init__(self):
        self.items = []

    def put(self, x, *a, **k):
        self.items.append(x)


class _FakeEnv:
    def close(self):
        pass


class _FakeTester:
    def __init__(self, *a, **k):
        self.env_list = [_FakeEnv()]


class _Cfg:
    graph_modalities = []
    pretrained_img_encoder_name = ""
    data_path = ""


def _run_worker(outcomes, task="CloseDrawer"):
    """Run env_worker_main in-process. `outcomes[rollout_idx]` is True/False (success)
    or "blank" (the rollout raises EmptyObservationError)."""
    orig = (env_worker._maybe_load_graph_encoders, env_worker.RoboCasaKitchenTester,
            env_worker._run_one_rollout)

    def fake_rollout(**kw):
        o = outcomes[kw["rollout_idx"]]
        if o == "blank":
            raise EmptyObservationError("no known object visible in agentview_left")
        return bool(o), {}

    env_worker._maybe_load_graph_encoders = lambda *a, **k: (None, None)
    env_worker.RoboCasaKitchenTester = _FakeTester
    env_worker._run_one_rollout = fake_rollout
    saved_stderr = os.dup(2)              # the worker redirects fd 2 to its log file
    done_q = _Q()
    try:
        with tempfile.TemporaryDirectory() as d:
            env_worker.env_worker_main(0, [(task, list(range(len(outcomes))))], _Cfg(),
                                       None, None, done_q, d, 42, False, 0, None)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        (env_worker._maybe_load_graph_encoders, env_worker.RoboCasaKitchenTester,
         env_worker._run_one_rollout) = orig
    return done_q.items


def test_isolated_blanks_are_scored_as_failures():
    outcomes = [True, "blank", False, "blank", "blank", True]
    msgs = _run_worker(outcomes)
    done = [m for m in msgs if isinstance(m, WorkerTaskDone)]
    assert len(done) == 1, f"expected one WorkerTaskDone, got {msgs}"
    assert done[0].per_scene == {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}, done[0].per_scene
    assert done[0].n_rollouts == 6 and done[0].n_success == 2
    assert not any(isinstance(m, WorkerError) for m in msgs), "an isolated blank killed the worker"


def test_a_run_of_blanks_flushes_and_exits():
    n = env_worker.MAX_CONSECUTIVE_BLANKS
    outcomes = [True, False, True] + ["blank"] * n + [True, True]
    msgs = _run_worker(outcomes)
    done = [m for m in msgs if isinstance(m, WorkerTaskDone)]
    assert len(done) == 1, f"partial results were not flushed exactly once: {msgs}"
    # Scored before the run survive; the whole blank run is un-scored for the retry.
    assert done[0].per_scene == {0: 1, 1: 0, 2: 1}, done[0].per_scene
    assert done[0].n_rollouts == 3 and done[0].n_success == 2
    assert any(isinstance(m, WorkerError) for m in msgs), "dead-renderer exit was not reported"
    assert isinstance(msgs[-1], WorkerDone), "WorkerDone must still be sent so the drain ends"


def test_a_completed_rollout_resets_the_blank_run():
    n = env_worker.MAX_CONSECUTIVE_BLANKS
    outcomes = ["blank"] * (n - 1) + [False] + ["blank"] * (n - 1)
    msgs = _run_worker(outcomes)
    assert not any(isinstance(m, WorkerError) for m in msgs), \
        "two runs of n-1 blanks separated by a completed rollout must not kill the worker"
    done = [m for m in msgs if isinstance(m, WorkerTaskDone)]
    assert len(done[0].per_scene) == len(outcomes)


# ------------------------------------------------------------------ 3. the tester
class _FakeServer:
    def __init__(self, zeros=0, err=None):
        self.zero_action_replies = zeros
        self.zero_action_first_error = err
        self.node_scores = NodeScoreTracker()


class _Scripted(parallel_tester.RoboCasaKitchenParallelTester):
    """`_run_round` replaced by a script: `lose[r]` is the set of (task, idx) rollouts
    that round r fails to report. Every reported rollout succeeds."""

    def __init__(self, lose, num_envs=4):
        super().__init__(num_envs=num_envs)
        self.lose = lose
        self.rounds = []

    def _run_round(self, assignments, method, manager, store_videos, working_dir,
                   test_count, epoch, per_task_success, per_task_count,
                   per_task_scene, per_task_faith, errors):
        r = len(self.rounds)
        self.rounds.append(assignments)
        lost = self.lose[r] if r < len(self.lose) else set()
        for worker in assignments:
            for task, idxs in worker:
                for i in idxs:
                    if (task, i) in lost:
                        continue
                    per_task_success[task] += 1
                    per_task_count[task] += 1
                    per_task_scene[task][i] = 1
        return _FakeServer(zeros=1, err=f"round {r}")


def _test(t, n=10, tasks=("CloseDrawer", "OpenDrawer")):
    return t.test(None, None, False, n, "", 0, None, task_names=list(tasks))


def test_lost_rollouts_are_retried_and_counted():
    t = _Scripted(lose=[{("CloseDrawer", 3), ("CloseDrawer", 7), ("OpenDrawer", 0)}])
    res = _test(t)
    assert len(t.rounds) == 2, f"expected one retry round, got {len(t.rounds) - 1}"
    retried = sorted((task, i) for w in t.rounds[1] for task, idxs in w for i in idxs)
    assert retried == [("CloseDrawer", 3), ("CloseDrawer", 7), ("OpenDrawer", 0)], retried
    assert len(t.rounds[1]) == 3, "a retry round should spawn only as many workers as needed"
    # Built AFTER the retries: every rollout counts.
    assert res == {"Avg_Test_Reward_CloseDrawer": 1.0, "Avg_Test_Reward_OpenDrawer": 1.0}, res
    assert all(len(v) == 10 for v in t.per_scene_success.values())


def test_a_round_that_recovers_nothing_stops_retrying():
    lost = {("CloseDrawer", 3)}
    t = _Scripted(lose=[lost, lost, lost])
    res = _test(t)
    assert len(t.rounds) == 2, f"kept retrying a dead device: {len(t.rounds)} rounds"
    assert res["Avg_Test_Reward_CloseDrawer"] == 0.9, res


def test_partial_recovery_uses_every_retry_round():
    t = _Scripted(lose=[{("CloseDrawer", 1), ("CloseDrawer", 2)}, {("CloseDrawer", 2)}])
    res = _test(t)
    assert len(t.rounds) == 3, len(t.rounds)
    assert res["Avg_Test_Reward_CloseDrawer"] == 1.0, res


def test_nothing_lost_means_no_retry():
    t = _Scripted(lose=[])
    _test(t)
    assert len(t.rounds) == 1


# ------------------------------------------------------------------ plots flag
def test_plots_flag_reaches_the_workers():
    """The worker only sees ManagerConfig, so the flag must survive _build_manager_cfg,
    and it must default to off for a manager that predates it."""
    import torch
    from types import SimpleNamespace
    base = dict(data_path="", use_proprioceptive=False, use_image=False, use_graph=True,
                prop_modalities=[], image_modalities=[], graph_modalities=[],
                y_bounds=torch.zeros(2, 12))
    on = parallel_tester._build_manager_cfg(
        SimpleNamespace(**base, store_sparsification_plots=True))
    old = parallel_tester._build_manager_cfg(SimpleNamespace(**base))
    assert on.store_sparsification_plots is True
    assert old.store_sparsification_plots is False


if __name__ == "__main__":
    tests = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = []
    for name in tests:
        try:
            globals()[name](); print(f"PASS  {name}")
        except AssertionError as e:
            failed.append(name); print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append(name); print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    sys.exit(0 if not failed else 1)
