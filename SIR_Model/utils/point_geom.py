"""Metric 3D geometry for graph nodes, derived from depth + segmentation.

Every graph node currently carries only appearance (a cropped-image embedding) and
2D pixel-space box coordinates, so the GNN has no idea where objects actually are in
space, how large they are, or how far the gripper is from them. This module computes a
compact per-node geometric descriptor from the object's reconstructed point cloud:

    segmentation mask + depth  ->  world-frame points  ->  robot-base frame  ->  15-d vector

Nothing here reads MuJoCo object state. The simulator is used only for the camera pose,
the robot base pose and the gripper pose — all three are available on a real robot from
calibrated extrinsics and forward kinematics — so the same features can be computed
outside simulation.

Layout of the 17-d descriptor (see `POINT_GEOM_DIM` / `POINT_GEOM_LAYOUT`):

     0:3   centroid of the object's points, robot-base frame
     3:9   orientation: the first two columns of the PCA rotation matrix
     9:12  extents: standard deviation along each principal axis (object size/shape)
    12:15  centroid - gripper position
    15:16  distance from the centroid to the gripper
    16:17  visibility: log1p(N_points) / log1p(POINT_GEOM_CONF_REF), 0 when unobserved

Orientation is stored as two axis vectors rather than a quaternion on purpose. A
quaternion double-covers SO(3) (q and -q are the same rotation), so the canonical-sign
convention flips whenever the pinned component crosses zero, and a re-sampled cloud of
the *same* object can produce two completely different 4-vectors — measured at up to 2.0
absolute deviation in `tests/test_point_geom.py`. Two columns of the rotation matrix are
continuous in the rotation, and the third is recoverable as their cross product.
"""
import numpy as np

from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)

# Width of the per-node descriptor. Mirrored in RoboCasa_Manager.calculate_graph_dim
# and dataloader.utils.GRAPH_MODALITY_SPEC.
POINT_GEOM_DIM = 17

POINT_GEOM_LAYOUT = (
    ("centroid", 3),
    ("rotation_6d", 6),
    ("extents", 3),
    ("gripper_offset", 3),
    ("gripper_distance", 1),
    ("visibility", 1),
)


def _offset_of(field):
    off = 0
    for name, width in POINT_GEOM_LAYOUT:
        if name == field:
            return off
        off += width
    raise KeyError(field)


# Offsets into the descriptor, so consumers slice by name instead of a magic index.
POINT_GEOM_CENTROID_OFFSET = _offset_of("centroid")
POINT_GEOM_VISIBILITY_OFFSET = _offset_of("visibility")

# Voxel resolution used to downsample every object cloud, in metres. The offline dataset
# pass and the rollout twin must both use this value: a denser cloud at evaluation time
# shifts centroids and extents, so the policy would see a different feature distribution
# than it was trained on.
POINT_GEOM_VOXEL_SIZE = 0.01

# Point count that saturates the visibility channel. Objects with this many points or
# more read as fully observed; the channel is a soft confidence, not a hard count.
POINT_GEOM_CONF_REF = 512

# Below this many points a PCA orientation is meaningless (a plane needs 3 points), so
# the node is reported as unobserved instead of carrying a fabricated pose.
MIN_POINTS_FOR_GEOMETRY = 3

# Body names to try when locating the robot base, in priority order.
_BASE_BODY_CANDIDATES = (
    "mobilebase0_root",
    "mobilebase0_base",
    "robot0_base",
    "base0",
    "base",
)

# The eef site body sits roughly at the tool centre point, which is what matters for
# "how far is the gripper from this object"; the wrist body is ~10 cm behind it.
_GRIPPER_BODY_CANDIDATES = (
    "gripper0_right_eef",
    "gripper0_eef",
    "gripper0_right_gripper",
    "gripper0_gripper",
    "robot0_right_hand",
    "robot0_hand",
)


# --------------------------------------------------------------------------------------
# Camera unprojection
# --------------------------------------------------------------------------------------

def unproject_mask_to_world(depth, mask, sim, camera_name):
    """Unproject the masked depth pixels of one object to world-frame XYZ.

    Args:
        depth: [H, W] or [H, W, 1] MuJoCo-normalized depth in [0, 1]
        mask:  [H, W] boolean mask selecting the pixels of one object
        sim:   robosuite MjSim
        camera_name: e.g. "robot0_agentview_left"

    Returns:
        np.ndarray [N, 3] world-frame points in metres; empty when no pixel is valid.
    """
    if depth.ndim == 3:
        depth = depth[..., 0]

    mask = np.asarray(mask).astype(bool)
    valid = mask & (depth > 0)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)

    H, W = depth.shape
    metric_depth = get_real_depth_map(sim, depth)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)
    d = metric_depth[valid].astype(np.float32)

    K = get_camera_intrinsic_matrix(sim=sim, camera_name=camera_name,
                                    camera_height=H, camera_width=W)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    pts_cam = np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d, np.ones_like(d)], axis=1)

    # The extrinsic already folds in the MuJoCo-vs-OpenCV axis correction.
    T_world_from_cam = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    return (T_world_from_cam @ pts_cam.T).T[:, :3].astype(np.float32)


def merge_views(*views):
    """Concatenate the per-view clouds of one object."""
    non_empty = [v for v in views if v is not None and v.shape[0] > 0]
    if not non_empty:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


def voxel_downsample(points, voxel_size):
    """Keep one representative point per occupied voxel.

    Equalizes the point density between near and far surfaces, so the centroid is not
    dragged towards whichever face happens to be closest to a camera.
    """
    if points.shape[0] == 0 or not voxel_size or voxel_size <= 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------

def _find_body(sim, candidates):
    """Return the id of the first candidate body present in the scene, else None."""
    names = {sim.model.body_id2name(i) for i in range(sim.model.nbody)}
    for cand in candidates:
        if cand in names:
            return sim.model.body_name2id(cand)
    return None


def get_base_pose(sim):
    """(base_pos[3], base_mat[3, 3]) of the robot base in world frame.

    Falls back to the world origin with identity rotation for a static scene with no
    recognisable mobile base, which makes the base frame equal the world frame.
    """
    bid = _find_body(sim, _BASE_BODY_CANDIDATES)
    if bid is None:
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)
    return (sim.data.body_xpos[bid].astype(np.float32),
            sim.data.body_xmat[bid].reshape(3, 3).astype(np.float32))


def get_gripper_pos(sim):
    """(pos[3], found) of the gripper in world frame."""
    bid = _find_body(sim, _GRIPPER_BODY_CANDIDATES)
    if bid is None:
        return np.zeros(3, dtype=np.float32), False
    return sim.data.body_xpos[bid].astype(np.float32), True


def world_to_base(points_world, base_pos, base_mat):
    """World-frame points to robot-base frame. base_mat is body->world, so it transposes.

    Working in the base frame makes the features invariant to where the robot happens to
    stand in the kitchen, which is what lets them transfer across scene layouts.
    """
    pts = np.atleast_2d(points_world)
    result = (pts - base_pos[None, :]) @ base_mat
    return result if np.ndim(points_world) > 1 else result[0]


def base_to_world(points_base, base_pos, base_mat):
    """Inverse of `world_to_base`; used by the diagnostics to check against sim state."""
    pts = np.atleast_2d(points_base)
    result = pts @ base_mat.T + base_pos[None, :]
    return result if np.ndim(points_base) > 1 else result[0]


# --------------------------------------------------------------------------------------
# Descriptor
# --------------------------------------------------------------------------------------

def rotation_to_6d(R):
    """[3, 3] rotation matrix -> [6] = its first two columns, stacked.

    The dropped third column is exactly `cross(col0, col1)`, so no information is lost.
    Unlike a quaternion this is continuous in the rotation, which matters because these
    features are consumed frame by frame by a network.
    """
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def rotation_from_6d(r6):
    """Inverse of `rotation_to_6d`: Gram-Schmidt the two axes back into a rotation."""
    a, b = np.asarray(r6[:3], dtype=np.float64), np.asarray(r6[3:], dtype=np.float64)
    c0 = a / max(np.linalg.norm(a), 1e-8)
    b = b - np.dot(c0, b) * c0
    c1 = b / max(np.linalg.norm(b), 1e-8)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1).astype(np.float32)


def pca_frame(points, gripper_pos_base):
    """Principal axes and per-axis spread of a point cloud.

    Returns (R[3, 3], extents[3]) where R's columns are the principal axes ordered by
    descending variance and extents are the standard deviations along them. The dominant
    axis is sign-fixed to point away from the gripper so the orientation is reproducible
    instead of flipping arbitrarily between frames (an eigenvector is only defined up to
    sign).
    """
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points), 1)

    # Trace-scaled regularization keeps eigenvalues separated for flat or near-collinear
    # clouds (a cabinet panel is essentially 2D), where eigh is otherwise ill-conditioned.
    trace = np.trace(cov)
    cov = cov + np.eye(3) * max(trace * 1e-6, 1e-9)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    order = np.argsort(eigvals)[::-1]              # descending variance
    eigvals = eigvals[order]
    R = eigvecs[:, order]

    # eigh returns each eigenvector with an arbitrary sign, so the raw axes are not
    # reproducible between frames. Both of the first two axes need pinning: fixing only
    # the dominant one leaves the second free to flip, which shows up as a full sign
    # inversion of half the descriptor.
    to_obj = points.mean(axis=0) - gripper_pos_base
    if np.dot(R[:, 0], to_obj) < 0:
        R[:, 0] = -R[:, 0]

    # Nothing physical orients the second axis, so pin it against a fixed reference.
    # Kitchen objects are gravity-aligned, so base +z is the most informative choice;
    # fall back through the other axes when the second axis is nearly perpendicular to
    # it and the sign would be decided by noise.
    for ref in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        d = float(np.dot(R[:, 1], ref))
        if abs(d) > 0.1:
            if d < 0:
                R[:, 1] = -R[:, 1]
            break

    # Third axis follows from the other two, which also guarantees a right-handed frame.
    R[:, 2] = np.cross(R[:, 0], R[:, 1])

    extents = np.sqrt(np.maximum(eigvals, 0.0))
    return R.astype(np.float32), extents.astype(np.float32)


def empty_point_geom():
    """Descriptor for a node with no usable points: zeros, identity rotation, visibility 0.

    Used for occluded objects. The graph carries nodes forward across frames even when
    they leave view, so the visibility channel is what tells the GNN that the rest of
    this vector is stale rather than measured.
    """
    feat = np.zeros(POINT_GEOM_DIM, dtype=np.float32)
    feat[3:9] = rotation_to_6d(np.eye(3))           # identity rotation
    return feat


def compute_point_geom(points_base, gripper_pos_base):
    """Reduce one object's robot-base-frame cloud to the 15-d descriptor.

    Args:
        points_base:      [N, 3] the object's points in robot-base frame
        gripper_pos_base: [3] gripper position in robot-base frame

    Returns:
        np.ndarray [POINT_GEOM_DIM] float32
    """
    points_base = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    if points_base.shape[0] < MIN_POINTS_FOR_GEOMETRY:
        return empty_point_geom()

    centroid = points_base.mean(axis=0)
    R, extents = pca_frame(points_base, np.asarray(gripper_pos_base, dtype=np.float32))

    offset = centroid - np.asarray(gripper_pos_base, dtype=np.float32)
    distance = np.linalg.norm(offset)
    visibility = np.log1p(points_base.shape[0]) / np.log1p(POINT_GEOM_CONF_REF)

    return np.concatenate([
        centroid,                       # 0:3
        rotation_to_6d(R),              # 3:9
        extents,                        # 9:12
        offset,                         # 12:15
        [distance],                     # 15:16
        [min(visibility, 1.0)],         # 16:17
    ]).astype(np.float32)


# Segmentation classes that belong to the robot rather than the scene.
ROBOT_CLASS_NAMES = ("PandaMobile", "PandaGripper", "OmronMobileBase")


def robot_class_ids(cls_list):
    """Segmentation class ids of the robot, given the env's `classes_to_ids` ordering.

    The class image encodes a class as its 1-based index in `env.model.classes_to_ids`,
    which is the same convention `RoboCasaKitchenTester.helper_function` uses.
    """
    names = list(cls_list.keys()) if isinstance(cls_list, dict) else list(cls_list)
    return {i + 1 for i, name in enumerate(names) if name in ROBOT_CLASS_NAMES}


def build_robot_mask(class_seg, robot_ids):
    """Boolean [H, W] mask of the pixels covered by the robot.

    RoboCasa occasionally assigns a robot geom an object class, and the gripper is by
    construction the thing closest to whatever is being manipulated, so leaving its
    pixels in would pull that object's centroid towards the hand. Built from the class
    segmentation, which is already rendered — no extra camera configuration needed.

    Returns None when there is nothing to mask, in which case callers skip the step.
    """
    if class_seg is None or not robot_ids:
        return None
    seg = np.asarray(class_seg)
    if seg.ndim == 3:
        seg = seg[..., 0]
    return np.isin(seg, list(robot_ids))


def project_points_to_image(points_world, sim, camera_name, height, width):
    """Project world-frame points into pixel coordinates of one camera.

    Used by the generator's `--verify` alignment check: a reconstructed centroid must
    land inside the object's recorded 2D bounding box, which fails loudly if the replay
    is off by a step from the recorded masks.

    Returns [N, 2] of (u, v) pixel coordinates.
    """
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float32))
    T_world_from_cam = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    T_cam_from_world = np.linalg.inv(T_world_from_cam)

    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (T_cam_from_world @ homog.T).T[:, :3]

    K = get_camera_intrinsic_matrix(sim=sim, camera_name=camera_name,
                                    camera_height=height, camera_width=width)
    z = np.clip(cam[:, 2], 1e-6, None)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def point_geom_features_from_views(sim, views, voxel_size=POINT_GEOM_VOXEL_SIZE):
    """Reconstruct every visible object and reduce it to its descriptor.

    This is the single implementation shared by the offline dataset pass
    (`utils.generate_point_geom_dataset`) and the rollout twin
    (`RoboCasaKitchenTester.get_point_geom_from_obs`). Keeping one copy is the point:
    if the two drifted — a different voxel size, a different frame — the policy would be
    evaluated on a feature distribution it never saw in training.

    Args:
        sim: robosuite MjSim, used for camera / base / gripper poses only
        views: iterable of (camera_name, depth[H, W], {object_name: mask[H, W]},
            robot_mask[H, W] or None)
        voxel_size: voxel-grid resolution in metres

    Returns:
        {object_name: np.ndarray [POINT_GEOM_DIM]} for every object seen in some view.
        Callers fill in `empty_point_geom()` for the nodes that are missing.
    """
    base_pos, base_mat = get_base_pose(sim)
    gripper_world, _ = get_gripper_pos(sim)
    gripper_base = world_to_base(gripper_world, base_pos, base_mat)

    per_object = {}
    for camera, depth, masks, robot_mask in views:
        if depth is None:
            continue
        for name, mask in masks.items():
            m = np.asarray(mask).astype(bool)
            if robot_mask is not None:
                m = m & ~robot_mask
            if not m.any():
                continue
            pts = unproject_mask_to_world(depth, m, sim, camera)
            if pts.shape[0]:
                per_object.setdefault(name, []).append(pts)

    features = {}
    for name, chunks in per_object.items():
        pts_base = world_to_base(merge_views(*chunks), base_pos, base_mat)
        features[name] = compute_point_geom(voxel_downsample(pts_base, voxel_size),
                                            gripper_base)
    return features


def stack_point_geom(features, node_names):
    """[len(node_names), POINT_GEOM_DIM] float32 tensor, in `node_names` order.

    Nodes absent from `features` were not observed this frame and get the unobserved
    descriptor, whose visibility channel is zero.
    """
    import torch

    if not node_names:
        return torch.zeros((0, POINT_GEOM_DIM), dtype=torch.float32)
    return torch.from_numpy(
        np.stack([features.get(n, empty_point_geom()) for n in node_names]).astype(np.float32)
    )


def centroid_slice(feat):
    """Extract the centroid block from a [..., POINT_GEOM_DIM] descriptor tensor/array."""
    o = POINT_GEOM_CENTROID_OFFSET
    return feat[..., o:o + 3]


def visibility_slice(feat):
    """Extract the visibility channel from a [..., POINT_GEOM_DIM] descriptor."""
    o = POINT_GEOM_VISIBILITY_OFFSET
    return feat[..., o:o + 1]
