"""Tests for per-node score tracking and the wandb metrics built from it.

These must be ordinary scalars, not a `wandb.plot.line_series` custom chart. A custom
chart renders as a static image — you cannot hover a line to see which object it is,
cannot zoom or smooth it, cannot compare one object across runs, and it never enters the
run history so the API cannot read it back. Scalars give all of that, and one panel
holding a whole task is then a matter of a metric regex in the UI.

The naming is load-bearing: `node_scores/<split>/<task>/<object>` is what lets a single
line panel pull in exactly one task's objects.

Run: python tests/test_node_scores.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.node_scores import NodeScoreTracker, wandb_log_dict


def _tracker(**by_mod_task_node):
    t = NodeScoreTracker()
    for key, (total, count) in by_mod_task_node.items():
        mod, task, name = key.split("__")
        t.sum[(mod, task, name)] = total
        t.cnt[(mod, task, name)] = count
    return t


def test_modalities_are_pooled_not_averaged_twice():
    """Two modalities seeing a node 6 and 4 times must give the count-weighted mean, not
    the mean of the two means."""
    t = _tracker(a__CloseDrawer__Drawer=(3.0, 6), b__CloseDrawer__Drawer=(1.0, 4))
    assert abs(t.task_node_means()[("CloseDrawer", "Drawer")] - 0.4) < 1e-9


def test_metric_names_group_by_split_and_task():
    """The regex `node_scores/val/CloseDrawer/.*` has to select that task's objects and
    nothing else, or a single panel cannot hold exactly one task."""
    t = _tracker(m__CloseDrawer__Drawer=(0.9, 1),
                 m__CloseDrawer__Counter=(0.5, 1),
                 m__CoffeePressButton__CoffeeMachine=(0.7, 1))
    keys = set(wandb_log_dict(t, "val", 0)) - {"node_epoch"}
    assert keys == {
        "node_scores/val/CloseDrawer/Drawer",
        "node_scores/val/CloseDrawer/Counter",
        "node_scores/val/CoffeePressButton/CoffeeMachine",
    }
    prefix = "node_scores/val/CloseDrawer/"
    assert sum(k.startswith(prefix) for k in keys) == 2


def test_values_are_plain_floats_not_chart_objects():
    """A chart object would also break the trainer's `f"{v:.6f}"` logging."""
    t = _tracker(m__CloseDrawer__Drawer=(0.9, 1))
    scores = wandb_log_dict(t, "val", 0)
    for k, v in scores.items():
        if k == "node_epoch":
            continue
        assert isinstance(v, float), f"expected a scalar, got {type(v).__name__}"
        assert f"{v:.6f}"


def test_the_axis_metric_is_emitted_alongside_the_scores():
    """Backfilled runs plot against `node_epoch` because a resumed run cannot reuse the
    global step. Live runs must emit the same axis or one panel cannot serve both."""
    t = _tracker(m__CloseDrawer__Drawer=(0.9, 1))
    assert wandb_log_dict(t, "val", 7)["node_epoch"] == 7
    assert "node_epoch" not in wandb_log_dict(NodeScoreTracker(), "val", 7)


def test_splits_do_not_collide():
    t = _tracker(m__T__a=(0.5, 1))
    assert set(wandb_log_dict(t, "train", 0)) - {"node_epoch"} == {"node_scores/train/T/a"}
    assert set(wandb_log_dict(t, "val", 0)) - {"node_epoch"} == {"node_scores/val/T/a"}


def test_empty_tracker_logs_nothing():
    assert wandb_log_dict(NodeScoreTracker(), "val", 0) == {}


def test_object_names_with_underscores_survive():
    """RoboCasa node names include `distr_counter_1` and `drawer_obj`; the metric path is
    slash-separated precisely so those stay readable."""
    t = _tracker(m__CloseDrawer__distr_counter_1=(0.3, 1))
    assert "node_scores/val/CloseDrawer/distr_counter_1" in wandb_log_dict(t, "val", 0)


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
