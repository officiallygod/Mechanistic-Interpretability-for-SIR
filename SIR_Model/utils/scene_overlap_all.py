"""Scene-novelty characterisation of the RoboCasa eval protocol, all 24 tasks.

Samples eval scenes at the REAL per-rollout seeds and asks, per task, how much of each
scene the 54 demonstrations of that task already contained. Renderer off -- this needs
ep_meta and mjModel only, so it runs beside a live eval without touching the GPU.

Writes robocasa_scene_novelty.json and prints a markdown table.
"""
import json
import os
import sys
import traceback

import h5py

from envs.robocasa.utils import TASK_LIST, create_kitchen_env, per_rollout_seed, seed_rollout

DATA = "/DATA/dataset"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE_SEED = 42
NO_RENDER = {"has_offscreen_renderer": False, "use_camera_obs": False,
             "camera_names": [], "camera_depths": False}


def demo_index(task):
    pairs, assets, configs = set(), set(), []
    with h5py.File(os.path.join(DATA, task, "demo_gentex_im128_randcams.hdf5"), "r") as f:
        for d in f["data"]:
            m = json.loads(f["data"][d].attrs["ep_meta"])
            pairs.add((m["layout_id"], m["style_id"]))
            a = sorted((c.get("info") or {})["mjcf_path"].rsplit("/model.xml")[0].split("/")[-1]
                       for c in (m.get("object_cfgs") or []) if (c.get("info") or {}).get("mjcf_path"))
            assets.update(a)
            configs.append(((m["layout_id"], m["style_id"]), tuple(a)))
    return pairs, assets, configs


def main():
    out = {}
    for task in TASK_LIST:
        try:
            pairs, assets, configs = demo_index(task)
            env = create_kitchen_env(f"{DATA}/", [task], use_depth=False,
                                     seed=BASE_SEED, env_overrides=NO_RENDER)[0][0]
            new_pair = new_asset = tot_asset = exact = 0
            for i in range(N):
                seed_rollout(env, per_rollout_seed(task, i, BASE_SEED))
                env.reset()
                m = env.get_ep_meta()
                p = (m.get("layout_id"), m.get("style_id"))
                a = sorted((c.get("info") or {})["mjcf_path"].rsplit("/model.xml")[0].split("/")[-1]
                           for c in (m.get("object_cfgs") or [])
                           if (c.get("info") or {}).get("mjcf_path"))
                if p not in pairs:
                    new_pair += 1
                tot_asset += len(a)
                new_asset += sum(1 for x in a if x not in assets)
                if (p, tuple(a)) in configs:
                    exact += 1
            out[task] = {"n": N, "demos": len(configs), "demo_pairs": len(pairs),
                         "demo_assets": len(assets), "unseen_pairs": new_pair,
                         "unseen_assets": new_asset, "total_assets": tot_asset,
                         "exact_config_repeats": exact}
            print(f"done {task}: pairs {new_pair}/{N} new, assets {new_asset}/{tot_asset} new, "
                  f"exact repeats {exact}/{N}", flush=True)
            del env
        except Exception:
            print(f"FAILED {task}\n{traceback.format_exc()}", flush=True)

    with open("robocasa_scene_novelty.json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n| task | demo (layout,style) pairs | unseen pair | unseen object instances | exact scene repeats |")
    print("|---|---|---|---|---|")
    tp = ta = tta = te = tn = 0
    for t, r in out.items():
        print(f"| {t} | {r['demo_pairs']}/100 | {100*r['unseen_pairs']/r['n']:.0f}% | "
              f"{100*r['unseen_assets']/max(r['total_assets'],1):.0f}% | {r['exact_config_repeats']}/{r['n']} |")
        tp += r["unseen_pairs"]; ta += r["unseen_assets"]; tta += r["total_assets"]
        te += r["exact_config_repeats"]; tn += r["n"]
    if tn:
        print(f"| **ALL 24** | — | **{100*tp/tn:.0f}%** | **{100*ta/max(tta,1):.0f}%** | **{te}/{tn}** |")


if __name__ == "__main__":
    main()
