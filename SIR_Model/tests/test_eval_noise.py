"""Deterministic eval noise: the diffusion starting noise is a function of where a
prediction sits in the evaluation -- (task, rollout_idx, step) -- not of the global RNG.

Without it the parallel path draws every prediction's noise from the policy server's
global RNG, so the noise a rollout receives depends on the order the workers' requests
arrived in, and re-running an unchanged evaluation disagrees scene by scene.

Run: python tests/test_eval_noise.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import torch

from envs.robocasa.utils import eval_noise

W, A, SIGMA_MAX = 10, 12, 80.0
M = SimpleNamespace(act_window_size=W, action_dim=A, sigma_max=SIGMA_MAX)


def test_same_key_same_noise_regardless_of_batch():
    alone = eval_noise(M, 42, [("CloseDrawer", 3, 40)])[0]
    batch = eval_noise(M, 42, [("OpenDrawer", 0, 0), ("CloseDrawer", 3, 40),
                               ("CloseDrawer", 4, 40)])
    swapped = eval_noise(M, 42, [("CloseDrawer", 4, 40), ("CloseDrawer", 3, 40)])
    assert torch.equal(alone, batch[1]), "noise depends on batch composition"
    assert torch.equal(alone, swapped[1]), "noise depends on batch position"
    assert torch.equal(alone, eval_noise(M, 42, [("CloseDrawer", 3, 40)])[0]), \
        "noise is not reproducible across calls"


def test_every_key_component_matters():
    base = eval_noise(M, 42, [("CloseDrawer", 3, 40)])[0]
    for seed, key in ((43, ("CloseDrawer", 3, 40)), (42, ("OpenDrawer", 3, 40)),
                      (42, ("CloseDrawer", 4, 40)), (42, ("CloseDrawer", 3, 50))):
        other = eval_noise(M, seed, [key])[0]
        assert not torch.equal(base, other), f"seed={seed} key={key} gave identical noise"


def test_shape_and_scale():
    n = eval_noise(M, 0, [("T", i, s) for i in range(20) for s in range(0, 500, 10)])
    assert n.shape == (1000, W, A), n.shape
    std = float(n.std())
    assert abs(std - SIGMA_MAX) / SIGMA_MAX < 0.02, f"std {std:.2f}, expected ~{SIGMA_MAX}"
    assert abs(float(n.mean())) < 0.02 * SIGMA_MAX


def test_independent_of_the_global_rng():
    torch.manual_seed(0); a = eval_noise(M, 42, [("T", 1, 2)])
    torch.manual_seed(999); torch.randn(1000); b = eval_noise(M, 42, [("T", 1, 2)])
    assert torch.equal(a, b), "eval_noise reads the global RNG"


def test_denoise_actions_starts_from_the_given_noise():
    from method.diffusion import Diffusion
    seen = {}
    fake = SimpleNamespace(
        num_sampling_steps=3, noise_scheduler="exponential", device="cpu",
        act_window_size=W, action_dim=A, sigma_max=SIGMA_MAX, sampler_type="ddim",
        get_noise_schedule=lambda n, s: torch.ones(n),
        sample_loop=lambda sig, x, st, g, t: seen.setdefault("x", x))
    noise = eval_noise(M, 42, [("T", 0, 0), ("T", 1, 0)])
    Diffusion.denoise_actions(fake, None, {"lang": torch.zeros(2, 4)}, noise=noise)
    assert torch.equal(seen["x"], noise), "denoise_actions did not start from the given noise"
    seen.clear()
    Diffusion.denoise_actions(fake, None, {"lang": torch.zeros(2, 4)})
    assert seen["x"].shape == (2, W, A), "default path changed shape"


def test_server_keys_on_rollout_not_on_worker():
    """The server must key on (task, rollout_idx, step). episode_id counts per worker, so
    keying on it would make the noise depend on num_envs and on retry rounds."""
    from envs.robocasa.parallel.policy_server import PolicyServer
    from envs.robocasa.parallel.protocol import PredictRequest

    seen = []

    class _Method:
        device = "cpu"
        act_window_size, action_dim, sigma_max = W, A, SIGMA_MAX

        def preprocess_batch(self, batch):
            return {}, None, batch["goal"]

        def predict(self, state, goal, noise=None):
            seen.append(noise)
            return torch.zeros(goal["lang"].shape[0], W, A)

    class _Q:
        def put(self, *a, **k):
            pass

    def run(items):
        srv = PolicyServer(_Method(), None, [_Q() for _ in range(8)], num_envs=8,
                           deterministic_noise=True, noise_seed=42)
        srv._forward_once(items)
        return seen[-1]

    def req(env_id, episode_id, rollout_idx, step):
        return PredictRequest(env_id=env_id, episode_id=episode_id, step=step,
                              obs={}, goal_lang=torch.zeros(1, 4),
                              task_name="CloseDrawer", rollout_idx=rollout_idx)

    a = run([req(0, 7, rollout_idx=5, step=30)])
    b = run([req(3, 3_000_002, rollout_idx=9, step=30),
             req(6, 6_000_001, rollout_idx=5, step=30)])     # other worker, other episode
    assert torch.equal(a[0], b[1]), "server noise depends on worker/episode, not the rollout"
    assert torch.equal(a[0], eval_noise(M, 42, [("CloseDrawer", 5, 30)])[0])


def test_result_files_record_the_sampler():
    from manager.robocasa_manager import RoboCasa_Manager
    on = SimpleNamespace(deterministic_eval_noise=True, eval_noise_seed=7)
    off = SimpleNamespace(deterministic_eval_noise=False, eval_noise_seed=7)
    assert RoboCasa_Manager._sampler_fields(on) == {"sampler": "seeded", "noise_seed": 7}
    assert RoboCasa_Manager._sampler_fields(off) == {"sampler": "legacy"}


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
