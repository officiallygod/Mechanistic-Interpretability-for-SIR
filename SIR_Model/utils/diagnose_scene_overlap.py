"""How different is an eval scene from the scenes in the 54 demonstrations?

Why
---
`create_kitchen_env` builds the eval env from the demo file's own `env_args`, so every
scene-generating parameter is inherited: `layout_ids=-1`, the same 10 `style_ids`,
`obj_instance_split='A'`, `generative_textures='100p'`, `randomize_cameras=True`. Rollout
scenes are therefore fresh draws from exactly the generator that produced the training
data -- new instantiations, same distribution.

This script measures that directly: it samples scenes with the real per-rollout seeds,
records what each one contains, and compares against every demo of that task.

Terminology, because it decides what the paper may claim:

  * A new draw from the SAME generator -- different object positions, different fixture
    placement, a layout/style pair this task never used -- is IN-distribution
    generalisation to unseen scene instantiations. It is not OOD, however visually
    different the scene looks. Distribution shift is a property of the sampling
    distribution, not of any individual sample.
  * `obj_instance_split='B'` IS a genuine held-out axis: it swaps in object meshes drawn
    from a disjoint pool, so the eval distribution differs from the training one. That is
    the only OOD lever available here -- pooled over all 24 tasks' 1293 demos, all 100
    (layout, style) pairs already appear in training.

Usage:
    python -m utils.diagnose_scene_overlap --task PnPCounterToSink --n 8
    python -m utils.diagnose_scene_overlap --task PnPCounterToSink --n 8 --split B
"""
import argparse
import json
import os

import h5py
import numpy as np

from envs.robocasa.utils import create_kitchen_env, per_rollout_seed, seed_rollout

DATA = "/DATA/dataset"


def demo_scenes(task):
    """{(layout,style)} and the object assets the demos actually used."""
    path = os.path.join(DATA, task, "demo_gentex_im128_randcams.hdf5")
    pairs, assets, positions = set(), set(), []
    with h5py.File(path, "r") as f:
        for d in f["data"]:
            m = json.loads(f["data"][d].attrs["ep_meta"])
            pairs.add((m["layout_id"], m["style_id"]))
            for c in (m.get("object_cfgs") or []):
                p = (c.get("info") or {}).get("mjcf_path")
                if p:
                    assets.add(p)
            positions.append(m)
    return pairs, assets, positions


def scene_record(env):
    """What this reset actually produced."""
    m = env.get_ep_meta()
    assets, cats = [], []
    for c in (m.get("object_cfgs") or []):
        info = c.get("info") or {}
        if info.get("mjcf_path"):
            assets.append(info["mjcf_path"])
            cats.append(info.get("cat"))
    # Object world positions: the part that makes each draw a distinct instantiation.
    pos = {}
    for name, obj in getattr(env, "objects", {}).items():
        try:
            pos[name] = np.round(
                env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)], 4).tolist()
        except Exception:
            pass
    return {"layout_id": m.get("layout_id"), "style_id": m.get("style_id"),
            "assets": assets, "cats": cats, "obj_pos": pos,
            "lang": m.get("lang")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCounterToSink")
    ap.add_argument("--n", type=int, default=8, help="rollout indices 0..n-1 to sample")
    ap.add_argument("--split", default=None, choices=["A", "B"],
                    help="obj_instance_split override; default = inherit from the dataset (A)")
    ap.add_argument("--base-seed", type=int, default=42)
    args = ap.parse_args()

    pairs_seen, assets_seen, demos = demo_scenes(args.task)
    print(f"=== {args.task}: {len(demos)} demos, {len(pairs_seen)} distinct (layout,style) "
          f"pairs, {len(assets_seen)} distinct object assets ===")

    # No renderer and no camera obs: this only needs ep_meta and mjModel, and keeping the
    # offscreen renderer off means it can run beside a live eval without fighting for GPU.
    overrides = {"has_offscreen_renderer": False, "use_camera_obs": False,
                 "camera_names": [], "camera_depths": False}
    if args.split:
        overrides["obj_instance_split"] = args.split
    env = create_kitchen_env(f"{DATA}/", [args.task], use_depth=False,
                             seed=args.base_seed, env_overrides=overrides)[0][0]

    recs = []
    for i in range(args.n):
        seed_rollout(env, per_rollout_seed(args.task, i, args.base_seed))
        env.reset()
        recs.append(scene_record(env))

    new_pairs = sum(1 for r in recs if (r["layout_id"], r["style_id"]) not in pairs_seen)
    all_a = [a for r in recs for a in r["assets"]]
    new_assets = [a for a in all_a if a not in assets_seen]

    print(f"\n{'#':>3} {'layout':>7}{'style':>6}  {'pair':<9} assets")
    for i, r in enumerate(recs):
        tag = "UNSEEN" if (r["layout_id"], r["style_id"]) not in pairs_seen else "seen"
        short = ",".join(a.split("/")[-2] for a in r["assets"])
        print(f"{i:>3} {r['layout_id']:>7}{r['style_id']:>6}  {tag:<9} {short[:70]}")

    print(f"\n(layout,style) pairs unseen for this task : {new_pairs}/{len(recs)}")
    print(f"object assets unseen in this task's demos  : {len(new_assets)}/{len(all_a)}"
          + (f"   e.g. {new_assets[0].split('/')[-2]}" if new_assets else ""))
    print(f"distinct scenes among the {len(recs)} samples: "
          f"{len({json.dumps(r['obj_pos'], sort_keys=True) for r in recs})}")

    exact = 0
    for r in recs:
        for m in demos:
            if (m["layout_id"], m["style_id"]) == (r["layout_id"], r["style_id"]):
                d_assets = sorted((c.get("info") or {}).get("mjcf_path")
                                  for c in (m.get("object_cfgs") or [])
                                  if (c.get("info") or {}).get("mjcf_path"))
                if d_assets == sorted(r["assets"]):
                    exact += 1
                    break
    print(f"samples reproducing a demo's (layout,style,assets) exactly: {exact}/{len(recs)}"
          f"   [positions still differ -- they are sampled continuously]")


if __name__ == "__main__":
    main()
