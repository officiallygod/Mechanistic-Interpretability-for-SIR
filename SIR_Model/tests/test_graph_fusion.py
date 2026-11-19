"""Tests for the left/right graph fusion and the geometric edge weights.

Run: python tests/test_graph_fusion.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch_geometric.data import Data

from dataloader.utils import (GRAPH_MODALITY_SPEC, build_edge_weights, fuse_graphs,
                              handle_fusing)
from utils.point_geom import (POINT_GEOM_CENTROID_OFFSET, POINT_GEOM_DIM,
                              POINT_GEOM_VISIBILITY_OFFSET, empty_point_geom)

CROP_W = 37


def _width(order, crop_w=CROP_W):
    return sum(crop_w if GRAPH_MODALITY_SPEC[m]["width"] is None
               else GRAPH_MODALITY_SPEC[m]["width"] for m in order)


def _fused_width(order, cropped_fusion, n=4):
    F = _width(order)
    left = torch.arange(n * F, dtype=torch.float32).reshape(n, F)
    fused, _ = handle_fusing(left, left + 1000.0, order, cropped_fusion)
    return fused.shape[-1]


# --------------------------------------------------------------------------- fusion

def test_widths_match_manager_formula():
    """These must equal RoboCasa_Manager.calculate_graph_dim or the model is built wrong."""
    assert _fused_width(["cropped_image_feature", "bb_coordinates"], True) == 37 + 10 * 2
    assert _fused_width(["cropped_image_feature", "bb_coordinates"], False) == 37 * 2 + 10 * 2
    assert _fused_width(["bb_coordinates", "cropped_image_feature"], True) == 57
    assert _fused_width(["one_hot_labels"], True) == 37
    assert _fused_width(["siglip_text_emb", "bb_coordinates"], True) == 768 + 20
    assert _fused_width(["point_geom"], True) == POINT_GEOM_DIM


def test_three_modalities_with_one_in_the_middle():
    """The previous implementation inferred each modality's position by string splitting,
    which raised ValueError as soon as a modality was neither first nor last."""
    order = ["one_hot_labels", "bb_coordinates", "cropped_image_feature"]
    assert _fused_width(order, True) == 37 + 20 + 37


def test_four_modalities_including_point_geom():
    order = ["one_hot_labels", "bb_coordinates", "point_geom", "cropped_image_feature"]
    assert _fused_width(order, True) == 37 + 20 + POINT_GEOM_DIM + 37


def test_offsets_locate_each_modality_in_the_output():
    order = ["one_hot_labels", "bb_coordinates", "point_geom"]
    F = _width(order)
    left = torch.zeros(2, F)
    fused, offsets = handle_fusing(left, left, order, True)
    assert offsets["one_hot_labels"] == (0, 37)
    assert offsets["bb_coordinates"] == (37, 20)        # concatenated across views
    assert offsets["point_geom"] == (57, POINT_GEOM_DIM)
    assert fused.shape[-1] == 57 + POINT_GEOM_DIM


def test_view_values_land_in_the_right_slots():
    order = ["one_hot_labels", "bb_coordinates"]
    left, right = torch.zeros(1, 47), torch.zeros(1, 47)
    left[0, :37] = 1.0;  right[0, :37] = 1.0            # view-invariant
    left[0, 37:] = 2.0;  right[0, 37:] = 3.0            # view-dependent
    fused, _ = handle_fusing(left, right, order, True)
    assert (fused[0, :37] == 1.0).all()
    assert (fused[0, 37:47] == 2.0).all(), "left bb block"
    assert (fused[0, 47:57] == 3.0).all(), "right bb block"


def test_right_only_node_falls_back_to_the_right_view():
    """A node only the right camera saw has an all-zero left row; a view-invariant
    feature must be read from the right view instead of emitting zeros."""
    order = ["one_hot_labels"]
    left, right = torch.zeros(2, 37), torch.zeros(2, 37)
    left[0, 5] = 1.0                                    # node 0 seen left only
    right[1, 9] = 1.0                                   # node 1 seen right only
    present_l = torch.tensor([True, False])
    present_r = torch.tensor([False, True])
    fused, _ = handle_fusing(left, right, order, True,
                             present_left=present_l, present_right=present_r)
    assert fused[0, 5] == 1.0, "left-only node lost its feature"
    assert fused[1, 9] == 1.0, "right-only node fell back to an empty left row"


def test_unknown_modality_is_rejected():
    try:
        handle_fusing(torch.zeros(1, 10), torch.zeros(1, 10), ["nope"], True)
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected a ValueError")


def test_width_mismatch_is_rejected():
    try:
        handle_fusing(torch.zeros(1, 99), torch.zeros(1, 99), ["one_hot_labels"], True)
    except ValueError as e:
        assert "99" in str(e)
    else:
        raise AssertionError("expected a ValueError")


# ---------------------------------------------------------------------- edge weights

def _point_geom_rows(centroids, observed=True):
    rows = []
    for c in centroids:
        f = empty_point_geom()
        f[POINT_GEOM_CENTROID_OFFSET:POINT_GEOM_CENTROID_OFFSET + 3] = c
        f[POINT_GEOM_VISIBILITY_OFFSET] = 1.0 if observed else 0.0
        rows.append(f)
    return torch.from_numpy(np.stack(rows))


def test_edge_weights_are_ones_without_point_geom():
    x = torch.zeros(3, 10)
    src = torch.tensor([0, 1, 2]); dst = torch.tensor([1, 2, 0])
    w = build_edge_weights(x, {"bb_coordinates": (0, 10)}, src, dst)
    assert torch.equal(w, torch.ones(3))


def test_edge_weights_decay_with_distance():
    x = _point_geom_rows([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    offsets = {"point_geom": (0, POINT_GEOM_DIM)}
    src = torch.tensor([0, 0]); dst = torch.tensor([1, 2])
    w = build_edge_weights(x, offsets, src, dst)
    assert torch.allclose(w, torch.tensor([1 / 2.0, 1 / 4.0]), atol=1e-6), w
    assert w[0] > w[1], "the nearer object must get the larger weight"


def test_edge_weights_are_symmetric():
    x = _point_geom_rows([[0., 0., 0.], [0.3, 0.4, 0.]])
    offsets = {"point_geom": (0, POINT_GEOM_DIM)}
    w = build_edge_weights(x, offsets, torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert torch.allclose(w[0], w[1])


def test_edge_weights_are_in_the_unit_interval():
    x = _point_geom_rows([[0., 0., 0.], [0.01, 0., 0.], [5., 5., 5.]])
    offsets = {"point_geom": (0, POINT_GEOM_DIM)}
    src = torch.tensor([0, 0, 1]); dst = torch.tensor([1, 2, 2])
    w = build_edge_weights(x, offsets, src, dst)
    assert bool((w > 0).all()) and bool((w <= 1).all()), w


def test_unobserved_nodes_fall_back_to_neutral_weight():
    """A zeroed centroid would otherwise fabricate a distance to the origin."""
    x = _point_geom_rows([[0., 0., 0.], [1., 0., 0.]])
    x[1, POINT_GEOM_VISIBILITY_OFFSET] = 0.0            # node 1 was never observed
    offsets = {"point_geom": (0, POINT_GEOM_DIM)}
    w = build_edge_weights(x, offsets, torch.tensor([0]), torch.tensor([1]))
    assert w.item() == 1.0, w


# ---------------------------------------------------------------------- end to end

def test_fuse_graphs_end_to_end_with_point_geom():
    order = ["bb_coordinates", "point_geom"]
    F = _width(order)

    def graph(names, centroids):
        x = torch.zeros(len(names), F)
        x[:, 10:] = _point_geom_rows(centroids)
        return Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0), node_names=names)

    left = {"m": graph(["Drawer", "obj"], [[0., 0., 0.], [0.5, 0., 0.]])}
    right = {"m": graph(["obj", "Counter"], [[0.5, 0., 0.], [2., 0., 0.]])}

    fused = fuse_graphs(left, right, "m", order, is_cropped_fusion=True)

    assert fused.node_names == ["Counter", "Drawer", "obj"], fused.node_names
    assert fused.x.shape == (3, 10 * 2 + POINT_GEOM_DIM)
    # Fully connected without self loops.
    assert fused.edge_index.shape == (2, 3 * 2)
    assert fused.edge_attr.shape == (6,)
    assert bool((fused.edge_attr > 0).all()) and bool((fused.edge_attr <= 1).all())

    # Drawer<->obj are 0.5 apart, Counter<->obj 1.5 apart, so the former binds tighter.
    idx = {n: i for i, n in enumerate(fused.node_names)}
    pair = {(int(s), int(d)): float(w) for (s, d), w in
            zip(fused.edge_index.t().tolist(), fused.edge_attr.tolist())}
    assert pair[(idx["Drawer"], idx["obj"])] > pair[(idx["Counter"], idx["obj"])]


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
