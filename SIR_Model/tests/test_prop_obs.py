"""Tests for the proprioceptive observation path, training-time and rollout-time.

Two bugs lived here at once, and each hid the other. `create_prop_obs` was a `pass`
stub, so any run with `use_proprioceptive` trained to completion and then died at the
first rollout on `obs[obs_prop]` being None — after the training cost was already paid.
And the width table in `calculate_proprioceptive_dim` disagreed with what
`get_proprioceptive_obs` actually concatenates (gripper_qpos 4 vs 2, joint_pos 3 vs 7,
`object` appended both raw and padded), so the model was built for a vector the
dataloader never produced.

The invariant worth pinning is that the manager's advertised width, the training
assembly and the rollout assembly all agree — nothing else catches a table that
drifts from the code it describes.

Run: python tests/test_prop_obs.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dataloader.datasets import OBJECT_OBS_DIM, RoboCasaDataset
from manager.robocasa_manager import RoboCasa_Manager

T = 5  # timesteps in the fake demo

# Measured off the demo HDF5s, not read back out of the table under test — the whole
# failure was a table that had drifted from the data it claimed to describe, so a test
# that sized its fixtures from that table would have passed just as happily.
DATASET_WIDTHS = {
    'robot0_eef_pos': 3,
    'robot0_eef_quat': 4,
    'robot0_base_pos': 3,
    'robot0_base_quat': 4,
    'robot0_base_to_eef_pos': 3,
    'robot0_base_to_eef_quat': 4,
    'robot0_gripper_qpos': 2,
    'robot0_gripper_qvel': 2,
    'robot0_joint_pos': 7,
    'robot0_joint_pos_cos': 7,
    'robot0_joint_pos_sin': 7,
    'robot0_joint_vel': 7,
}
# `object` is variable (14 per sampled object): 14 on the Coffee/Stove tasks, up to 56
# on the door/drawer ones, and it varies between demos of a single task too.
DATASET_OBJECT_WIDTH_MAX = 56


def _fake_demo(mods, object_width=28):
    """Stand-in for one demo's `obs` group: h5py datasets index the same way."""
    demo = {}
    for m in mods:
        width = object_width if m == 'object' else DATASET_WIDTHS[m]
        demo[m] = np.arange(T * width, dtype=np.float64).reshape(T, width)
    return demo


class _FakeH5(dict):
    """`demo_data[m][()]` is how the real code reads an h5py dataset."""

    def __getitem__(self, k):
        return _FakeArray(dict.__getitem__(self, k))


class _FakeArray:
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, key):
        return self.arr


def _train_obs(mods, object_width=28):
    demo = _FakeH5(_fake_demo(mods, object_width))
    return RoboCasaDataset.get_proprioceptive_obs(None, demo, mods)


class _FakeManager:
    def __init__(self, mods):
        self.prop_modalities = mods


def _rollout_obs(mods, object_width=28):
    from envs.robocasa.kitchen import RoboCasaKitchenTester
    obs_all = {m: np.arange(object_width if m == 'object' else DATASET_WIDTHS[m],
                            dtype=np.float64)
               for m in mods}
    return RoboCasaKitchenTester.create_prop_obs(None, obs_all, _FakeManager(mods))


# ------------------------------------------------------- the three widths must agree

def test_table_matches_the_widths_in_the_dataset():
    """The bug in its purest form: gripper_qpos was declared 4 (it is 2) and joint_pos 3
    (it is 7), so a config naming either built the model for the wrong vector."""
    for mod, width in DATASET_WIDTHS.items():
        assert RoboCasa_Manager.PROP_MODALITY_DIMS[mod] == width, \
            f"{mod}: table says {RoboCasa_Manager.PROP_MODALITY_DIMS[mod]}, data has {width}"
    assert RoboCasa_Manager.PROP_MODALITY_DIMS['object'] == DATASET_OBJECT_WIDTH_MAX
    assert OBJECT_OBS_DIM == DATASET_OBJECT_WIDTH_MAX


def test_declared_width_matches_the_training_assembly():
    for mods in (['robot0_eef_pos', 'robot0_eef_quat'],
                 ['robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos'],
                 ['robot0_joint_pos', 'robot0_joint_vel', 'robot0_gripper_qpos'],
                 ['robot0_eef_pos', 'object']):
        declared = RoboCasa_Manager.calculate_proprioceptive_dim(RoboCasa_Manager, mods)
        actual = _train_obs(mods).shape[-1]
        assert declared == actual, f"{mods}: manager says {declared}, dataset builds {actual}"


def test_rollout_assembly_matches_the_training_assembly():
    """A policy must see the same layout at rollout as it did in training."""
    mods = ['robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos']
    train = _train_obs(mods)
    roll = _rollout_obs(mods)
    assert roll.shape == (1, 1, train.shape[-1])
    # Step 0 of the fake demo is arange(width) per modality, same as the fake env step.
    assert torch.allclose(roll.reshape(-1), train[0])


def test_the_widest_object_block_still_fits_its_budget():
    """OBJECT_OBS_DIM is a pad target, so a task wider than it would be truncated."""
    obs = _train_obs(['object'], object_width=DATASET_OBJECT_WIDTH_MAX)
    assert obs.shape == (T, OBJECT_OBS_DIM)


def test_object_is_padded_once_not_appended_twice():
    """The raw `object` block used to be appended alongside its padded copy, which
    doubled the width and silently desynced the model from the data."""
    mods = ['robot0_eef_pos', 'object']
    assert _train_obs(mods, object_width=28).shape == (T, 3 + OBJECT_OBS_DIM)
    assert _rollout_obs(mods, object_width=28).shape == (1, 1, 3 + OBJECT_OBS_DIM)


def test_object_padding_keeps_the_values_and_zeroes_the_tail():
    obs = _train_obs(['object'], object_width=28)
    assert torch.allclose(obs[0, :28], torch.arange(28, dtype=torch.float32))
    assert bool((obs[:, 28:] == 0).all())


# ------------------------------------------------------------------ failure modes

def test_unknown_modality_is_rejected_by_the_manager():
    try:
        RoboCasa_Manager.calculate_proprioceptive_dim(RoboCasa_Manager, ['robot0_nonsense'])
    except ValueError:
        return
    raise AssertionError("unknown proprioceptive modality was accepted")


def test_modality_the_live_env_does_not_expose_fails_loudly():
    """`object` exists in the demo HDF5s but not in the live env, so a run configured
    with it would otherwise fail obscurely on a None deep inside the batcher."""
    from envs.robocasa.kitchen import RoboCasaKitchenTester
    obs_all = {'robot0_eef_pos': np.zeros(3)}
    try:
        RoboCasaKitchenTester.create_prop_obs(
            None, obs_all, _FakeManager(['robot0_eef_pos', 'object']))
    except KeyError as e:
        assert 'object' in str(e)
        return
    raise AssertionError("a missing live-env modality was silently accepted")


if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    failed = []
    for name in tests:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    sys.exit(0 if not failed else 1)
