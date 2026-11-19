"""Tests for the point_geom node descriptor.

Run: python tests/test_point_geom.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.point_geom import (
    POINT_GEOM_DIM, POINT_GEOM_LAYOUT,
    base_to_world, compute_point_geom, empty_point_geom, pca_frame,
    rotation_from_6d, rotation_to_6d, voxel_downsample, world_to_base,
)


GRIPPER = np.array([0.5, 2.0, 0.5], dtype=np.float32)


def _box_cloud(n=500, size=(0.30, 0.10, 0.05), center=(1.0, 2.0, 0.5), seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, (n, 3)) * np.array(size) + np.array(center)).astype(np.float32)


def test_layout_matches_dim():
    assert sum(w for _, w in POINT_GEOM_LAYOUT) == POINT_GEOM_DIM


def test_descriptor_recovers_box_geometry():
    f = compute_point_geom(_box_cloud(), GRIPPER)
    assert f.shape == (POINT_GEOM_DIM,), f.shape
    assert np.allclose(f[0:3], [1.0, 2.0, 0.5], atol=0.05), f[0:3]
    # Extents are the per-axis standard deviations, sorted descending.
    assert f[9] > f[10] > f[11], f[9:12]
    assert np.allclose(f[9:12], [0.30, 0.10, 0.05], atol=0.03), f[9:12]
    assert np.allclose(f[12:15], [0.5, 0.0, 0.0], atol=0.05), f[12:15]
    assert abs(f[15] - 0.5) < 0.05, f[15]
    assert 0.0 < f[16] <= 1.0, f[16]


def test_gripper_distance_tracks_approach():
    """The channel the policy most needs: distance shrinks as the gripper closes in."""
    pts = _box_cloud()
    dists = [compute_point_geom(pts, np.array([x, 2.0, 0.5], dtype=np.float32))[15]
             for x in (0.0, 0.25, 0.5, 0.75, 0.95)]
    assert all(b < a for a, b in zip(dists, dists[1:])), dists


def test_too_few_points_is_reported_as_unobserved():
    f = compute_point_geom(_box_cloud(n=2), GRIPPER)
    assert np.allclose(f, empty_point_geom())
    assert f[16] == 0.0, "visibility must be zero for an unobserved node"
    assert np.allclose(rotation_from_6d(f[3:9]), np.eye(3), atol=1e-5)


def test_orientation_is_stable_under_resampling():
    """The same object seen with a different point subset must give the same orientation.

    A quaternion fails this: q and -q are the same rotation, so the canonical-sign
    convention flips when the pinned component crosses zero, producing deviations up to
    2.0 for identical geometry. The 6D representation is continuous.
    """
    rng = np.random.default_rng(1)
    pts = _box_cloud()
    rots = np.stack([compute_point_geom(pts[rng.permutation(len(pts))[:400]], GRIPPER)[3:9]
                     for _ in range(16)])
    assert np.abs(rots - rots[0]).max() < 0.05, np.abs(rots - rots[0]).max()


def test_orientation_axes_are_orthonormal():
    f = compute_point_geom(_box_cloud(), GRIPPER)
    R = rotation_from_6d(f[3:9])
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5)
    assert abs(np.linalg.det(R) - 1.0) < 1e-5, "must be a rotation, not a reflection"


def test_rotation_6d_round_trip():
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 2] = -Q[:, 2]
    assert np.allclose(rotation_from_6d(rotation_to_6d(Q)), Q, atol=1e-5)


def test_dominant_axis_points_away_from_gripper():
    """Sign-fixing against the gripper is what makes the frame reproducible."""
    pts = _box_cloud()
    for gx in (0.0, 2.0):                       # gripper on either side of the object
        g = np.array([gx, 2.0, 0.5], dtype=np.float32)
        R, _ = pca_frame(pts, g)
        assert np.dot(R[:, 0], pts.mean(axis=0) - g) >= 0


def test_flat_cloud_does_not_blow_up():
    """Cabinet panels and doors are essentially 2D; eigh is ill-conditioned there."""
    rng = np.random.default_rng(3)
    flat = (rng.normal(0, 1, (300, 3)) * np.array([0.4, 0.3, 0.0])).astype(np.float32)
    f = compute_point_geom(flat, GRIPPER)
    assert np.isfinite(f).all(), f
    assert abs(f[11]) < 1e-3, "the degenerate axis should have ~zero extent"


def test_collinear_cloud_does_not_blow_up():
    rng = np.random.default_rng(4)
    t = rng.uniform(-1, 1, (200, 1)).astype(np.float32)
    line = (t * np.array([0.2, 0.1, 0.05])).astype(np.float32)
    assert np.isfinite(compute_point_geom(line, GRIPPER)).all()


def test_features_are_invariant_to_base_placement():
    """Working in the base frame is what makes these transfer across scene layouts."""
    pts = _box_cloud()
    f_ref = compute_point_geom(pts, GRIPPER)

    th = 0.7
    base_pos = np.array([0.2, -0.3, 0.1], dtype=np.float32)
    base_mat = np.array([[np.cos(th), -np.sin(th), 0.0],
                         [np.sin(th), np.cos(th), 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
    # Move the whole scene (object + gripper) with the base; base-frame features must not move.
    pts_moved = base_to_world(pts, base_pos, base_mat)
    grip_moved = base_to_world(GRIPPER, base_pos, base_mat)
    f_moved = compute_point_geom(world_to_base(pts_moved, base_pos, base_mat),
                                 world_to_base(grip_moved, base_pos, base_mat))
    assert np.allclose(f_ref, f_moved, atol=1e-4), np.abs(f_ref - f_moved).max()


def test_world_base_round_trip():
    pts = _box_cloud()
    bp = np.array([0.2, -0.3, 0.1], dtype=np.float32)
    th = 0.7
    bm = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th), np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    assert np.allclose(base_to_world(world_to_base(pts, bp, bm), bp, bm), pts, atol=1e-4)


def test_voxel_downsample_reduces_and_preserves_extent():
    pts = _box_cloud(n=5000)
    down = voxel_downsample(pts, 0.02)
    assert 0 < len(down) < len(pts)
    assert np.allclose(down.min(axis=0), pts.min(axis=0), atol=0.05)
    assert np.allclose(down.max(axis=0), pts.max(axis=0), atol=0.05)
    assert len(voxel_downsample(pts, 0)) == len(pts), "voxel_size 0 must be a no-op"
    assert len(voxel_downsample(np.zeros((0, 3), np.float32), 0.02)) == 0


def test_voxel_downsample_stabilises_the_centroid():
    """Density equalization is the point: a camera-facing face must not drag the centroid."""
    rng = np.random.default_rng(5)
    sparse_face = rng.uniform(-0.1, 0.1, (50, 3)).astype(np.float32) + np.array([0.2, 0, 0])
    dense_face = rng.uniform(-0.1, 0.1, (5000, 3)).astype(np.float32) + np.array([-0.2, 0, 0])
    cloud = np.concatenate([sparse_face, dense_face])
    raw_bias = abs(cloud.mean(axis=0)[0])
    down_bias = abs(voxel_downsample(cloud, 0.02).mean(axis=0)[0])
    assert down_bias < raw_bias, (raw_bias, down_bias)


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
