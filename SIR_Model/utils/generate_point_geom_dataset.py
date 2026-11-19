"""Build the `point_geom` graph modality by replaying demos with depth enabled.

The other graph modalities are derived from `dataset_raw.hdf5`, which stores per-object
bounding boxes and segmentation masks but no depth. Metric 3D geometry needs depth, so
this script replays each demo in the simulator with the depth cameras switched on, then
reduces every object's reconstructed point cloud to the descriptor in `utils.point_geom`.

Object masks are read straight out of `dataset_raw.hdf5` rather than recomputed, which
guarantees this modality has exactly the same nodes, in the same order, as the modalities
it will be concatenated with — `combine_graph_modalities` assumes that alignment and
would otherwise silently mix features from different objects.

Writes, alongside the other modality files:
    <data_path>/<task>/point_geom_left_image.pth
    <data_path>/<task>/point_geom_right_image.pth
each a list[demo_idx][step_idx] -> torch_geometric.data.Data.

Run:
    python -m utils.generate_point_geom_dataset --task CloseDrawer
    python -m utils.generate_point_geom_dataset --task ALL
"""
import argparse
import os

import h5py
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from envs.robocasa.utils import TASK_LIST, create_kitchen_env
from utils.create_graphs import create_graph_datapoint
from utils.point_geom import (
    POINT_GEOM_CENTROID_OFFSET, POINT_GEOM_VISIBILITY_OFFSET, POINT_GEOM_VOXEL_SIZE,
    base_to_world, build_robot_mask, get_base_pose, point_geom_features_from_views,
    project_points_to_image, robot_class_ids, stack_point_geom,
)

VIEWS = ("left", "right")
CAMERA_OF_VIEW = {"left": "robot0_agentview_left", "right": "robot0_agentview_right"}


def _object_masks_at_step(step_group, view):
    """{object_name: mask[H, W] bool} for one view at one step, from dataset_raw.hdf5.

    Uses the `_MorphMask` entries, which are the same masks the cropped-image features
    are built from, so the node sets line up.
    """
    group = step_group[f"{view}_image"]
    masks = {}
    for key in group.keys():
        if not key.endswith("_MorphMask"):
            continue
        name = key[: -len("_MorphMask")]
        masks[name] = np.asarray(group[key][()]).astype(bool)
    return masks


def _node_names_at_step(step_group, view):
    """Node names for one view at one step, in the order the other modalities use."""
    group = step_group[f"{view}_image"]
    return [k[: -len("_MorphMask")] for k in group.keys() if k.endswith("_MorphMask")]


def _bounding_boxes_at_step(step_group, view):
    """{object_name: (x_min, y_min, x_max, y_max)} recorded in dataset_raw.hdf5."""
    group = step_group[f"{view}_image"]
    return {k[: -len("_bb")]: np.asarray(group[k][()]) for k in group.keys()
            if k.endswith("_bb")}


def _check_alignment(features, step_group, sim, camera, view, base_pos, base_mat,
                     height, width):
    """Reproject each centroid and check it lands inside the recorded 2D box.

    This is the guard against the one thing that would corrupt the whole dataset without
    any visible error: the simulator replay being off by a step from the masks recorded
    in the hdf5. A time-misaligned centroid reprojects outside the object's own box.

    Returns (n_checked, n_inside).
    """
    boxes = _bounding_boxes_at_step(step_group, view)
    checked = inside = 0
    for name, feat in features.items():
        box = boxes.get(name)
        if box is None or feat[POINT_GEOM_VISIBILITY_OFFSET] <= 0:
            continue
        centroid_base = feat[POINT_GEOM_CENTROID_OFFSET:POINT_GEOM_CENTROID_OFFSET + 3]
        world = base_to_world(centroid_base, base_pos, base_mat)
        u, v = project_points_to_image(world, sim, camera, height, width)[0]
        x_min, y_min, x_max, y_max = box
        # A one-pixel margin: the centroid of a partially occluded surface can sit
        # marginally outside a tight box without anything being wrong.
        checked += 1
        if (x_min - 1) <= u <= (x_max + 1) and (y_min - 1) <= v <= (y_max + 1):
            inside += 1
    return checked, inside


def _reset_to_state(env, state):
    """Restore the simulator to a recorded state and refresh the renderer."""
    env.sim.set_state_from_flattened(state)
    env.sim.forward()


def generate_for_task(data_path, task_name, voxel_size=POINT_GEOM_VOXEL_SIZE, seed=0,
                      max_demos=None, verify=False):
    task_dir = os.path.join(data_path, task_name)
    raw_path = os.path.join(task_dir, "dataset_raw.hdf5")
    demo_path = os.path.join(task_dir, "demo_gentex_im128_randcams.hdf5")

    for p in (raw_path, demo_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file is missing: {p}")

    env_list, _ = create_kitchen_env(task_dir + "/", [task_name], use_depth=True, seed=seed)
    env = env_list[0]
    env.reset()
    robot_ids = robot_class_ids(env.model.classes_to_ids)
    align_checked = align_inside = 0

    raw = h5py.File(raw_path, "r")
    demos = h5py.File(demo_path, "r")

    # Same demo ordering as create_graphs_and_save, so index i lines up across modalities.
    raw_keys = np.array(raw)
    order = np.argsort(np.array([int(k.split("_", 1)[-1]) for k in raw_keys]))
    raw_keys = raw_keys[order]
    if max_demos is not None:
        raw_keys = raw_keys[:max_demos]

    all_data = {view: [] for view in VIEWS}

    for key in tqdm(raw_keys, desc=f"{task_name} demos"):
        graphs = {view: nx.DiGraph() for view in VIEWS}
        per_demo = {view: [] for view in VIEWS}

        states = demos["data"][key]["states"][()]

        step_keys = np.array(raw[key])
        num_steps = int(max(int(s) for s in step_keys)) + 1

        if len(states) < num_steps:
            print(f"WARNING {key}: {len(states)} recorded states for {num_steps} steps; "
                  f"the trailing steps reuse the last available state")

        for j in range(num_steps):
            step_group = raw[key][str(j)]

            if j < len(states):
                _reset_to_state(env, states[j])
            obs = env._get_observations(force_update=True)

            # Reconstruct each object once from both views; the descriptor is
            # view-invariant, so the same value is written into both view files.
            view_inputs = []
            for view in VIEWS:
                camera = CAMERA_OF_VIEW[view]
                robot_mask = build_robot_mask(obs.get(f"{camera}_segmentation_class"),
                                              robot_ids)
                view_inputs.append((camera, obs.get(f"{camera}_depth"),
                                    _object_masks_at_step(step_group, view), robot_mask))

            features = point_geom_features_from_views(env.sim, view_inputs, voxel_size)

            if verify:
                depth = np.asarray(obs[f"{CAMERA_OF_VIEW['left']}_depth"])
                base_pos, base_mat = get_base_pose(env.sim)
                c, i = _check_alignment(features, step_group, env.sim,
                                        CAMERA_OF_VIEW["left"], "left",
                                        base_pos, base_mat, depth.shape[0], depth.shape[1])
                align_checked += c
                align_inside += i

            for view in VIEWS:
                names = _node_names_at_step(step_group, view)
                per_demo[view].append(
                    create_graph_datapoint(graphs[view], names,
                                           stack_point_geom(features, names)))

        for view in VIEWS:
            all_data[view].append(per_demo[view])

    raw.close()
    demos.close()
    env.close()

    if verify and align_checked:
        rate = align_inside / align_checked
        print(f"alignment check: {align_inside}/{align_checked} centroids "
              f"({rate:.1%}) reproject inside their recorded bounding box")
        if rate < 0.9:
            print("WARNING: a low rate here usually means the simulator replay is "
                  "misaligned with the steps recorded in dataset_raw.hdf5, which would "
                  "attach each object's geometry to the wrong frame.")

    for view in VIEWS:
        out = os.path.join(task_dir, f"point_geom_{view}_image.pth")
        torch.save(all_data[view], out)
        print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True,
                        help="Task name, or ALL for every task in TASK_LIST")
    parser.add_argument("--data-path", default=None,
                        help="Dataset root; defaults to $SIR_DATA_PATH")
    parser.add_argument("--voxel-size", type=float, default=POINT_GEOM_VOXEL_SIZE,
                        help="Voxel resolution in metres. Must match the rollout twin "
                             "in envs/robocasa/kitchen.py")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-demos", type=int, default=None,
                        help="Only process the first N demos (for a quick check)")
    parser.add_argument("--verify", action="store_true",
                        help="Reproject each centroid and report how often it lands in "
                             "the object's recorded 2D box. Run this once per task with "
                             "--max-demos 2 before generating everything")
    args = parser.parse_args()

    data_path = args.data_path or os.environ.get("SIR_DATA_PATH")
    if not data_path:
        raise SystemExit("Pass --data-path or set SIR_DATA_PATH")

    tasks = TASK_LIST if args.task == "ALL" else [args.task]
    for task in tasks:
        print(f"=== {task} ===")
        generate_for_task(data_path, task, voxel_size=args.voxel_size, seed=args.seed,
                          max_demos=args.max_demos, verify=args.verify)


if __name__ == "__main__":
    main()
