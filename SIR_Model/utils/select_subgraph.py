"""Cut a stage-1 node-score snapshot into a fixed stage-2 subgraph.

Usage (from the repo root):
    python -m utils.select_subgraph <snapshot.json> --threshold 0.6 -o subgraph.yaml

Reads one snapshot written by NodeScoreTracker.save_json (e.g.
node_scores/val_epoch049.json from the stage-1 soft VGIB run), averages each
node's mean score across graph modalities, and keeps every node scoring above
one task-independent `--threshold`. Prints the ranked table per task so the cut
is visible, then writes the whitelist consumed by
`method.graph_encoder.sparsification_layer.node_whitelist` in stage 2.

`--task-threshold TaskName=0.8` (repeatable) overrides the cutoff for one task;
stage 2 reads the whitelist per task, so the cutoffs need not agree. Note that
per-task cutoffs make subgraph sizes vary by task, so a cross-task comparison of
stage-2 results is no longer size-matched.

Run it without --threshold first: the tables and the per-cutoff keep-counts show
where the gaps are, and nothing is written.
"""
import argparse
from datetime import datetime

import yaml

from utils.node_scores import load_snapshot

try:
    from utils.generate_graph_dataset_robocasa import RELEVENT_NODES
except ImportError:
    RELEVENT_NODES = None

SCAN_CUTOFFS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

# Trivially relevant to every task, so they carry no information about WHICH task this is.
# Under --scene-kmeans they are removed before the cut is fitted and re-added afterwards;
# `RELEVENT_NODES` names exactly PandaMobile and PandaGripper on all 24 tasks and never
# the mobile base, so that is what gets re-added.
ROBOT_NODES = ("PandaMobile", "PandaGripper", "OmronMobileBase")
ROBOT_KEPT = ("PandaMobile", "PandaGripper")


def pool_means(snapshot, modality=None):
    """{task: {node name: mean score averaged over modalities}}."""
    scores = snapshot.get("scores", {})
    if modality:
        scores = {m: t for m, t in scores.items() if m == modality}
        if not scores:
            raise SystemExit(f"modality '{modality}' not in snapshot "
                             f"(available: {list(snapshot.get('scores', {}))})")
    per_mod = {}
    for _, tasks in sorted(scores.items()):
        for task, entry in tasks.items():
            for name, stats in entry["nodes"].items():
                per_mod.setdefault(task, {}).setdefault(name, []).append(stats["mean"])
    return {task: {name: sum(v) / len(v) for name, v in names.items()}
            for task, names in per_mod.items()}


def pooled_std(snapshot, modality=None):
    """{task: {node name: max std across modalities}} — flicker risk near the bar."""
    scores = snapshot.get("scores", {})
    if modality:
        scores = {m: t for m, t in scores.items() if m == modality}
    out = {}
    for _, tasks in sorted(scores.items()):
        for task, entry in tasks.items():
            for name, stats in entry["nodes"].items():
                prev = out.setdefault(task, {}).get(name, 0.0)
                out[task][name] = max(prev, stats["std"])
    return out



def kmeans_split(ranked):
    """1D 2-means on one task's node scores. Returns (kept names, threshold).

    Seeds the "relevant" cluster A with the top-2 nodes and the rest into B, then
    repeatedly moves B's highest scorer into A while it sits closer to A's mean than
    to B's, recomputing both means each time. Fully unsupervised: no target size, no
    hand-set cutoff, nothing task-specific.

    The returned threshold is the MIDPOINT of the gap the split opened (between A's
    lowest and B's highest). Eval re-scores every node at every rollout step, so those
    live scores differ from the snapshot means this was fitted on; sitting in the middle
    of the gap is the maximum-margin choice and the one least likely to flip a node.
    """
    names = [n for n, _ in ranked]
    vals = [v for _, v in ranked]
    if len(vals) <= 2:
        return list(names), 0.0                      # nothing to split; keep all

    k = 2
    while k < len(vals):
        mean_a = sum(vals[:k]) / k
        mean_b = sum(vals[k:]) / (len(vals) - k)
        x = vals[k]                                  # B's highest, list is sorted desc
        if abs(x - mean_a) < abs(x - mean_b):
            k += 1
        else:
            break

    # A cut is only meaningful between two DISTINCT scores. On a tie the midpoint lands
    # exactly on the tied value, and since eval re-scores every node each rollout step,
    # which side of the bar a tied node falls on is then a coin flip. Walk past the run
    # of equal values; if that consumes the list the scores carry no split at all, so
    # keep everything and let the size warning say so rather than invent a subgraph.
    while k < len(vals) and vals[k] == vals[k - 1]:
        k += 1
    if k >= len(vals):
        return list(names), 0.0
    threshold = (vals[k - 1] + vals[k]) / 2.0
    return names[:k], threshold

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshot", help="stage-1 node_scores/*.json snapshot")
    parser.add_argument("--threshold", type=float,
                        help="task-independent cutoff; nodes with mean score > threshold are kept")
    parser.add_argument("--task-threshold", action="append", default=[], metavar="TASK=VALUE",
                        help="per-task cutoff overriding --threshold, e.g. OpenSingleDoor=0.8 "
                             "(repeatable)")
    parser.add_argument("--kmeans", action="store_true",
                        help="pick each task's subgraph by 1D 2-means on its own scores "
                             "instead of a cutoff; unsupervised, no per-task tuning. "
                             "Derives a per-task threshold too, for eval-time pruning.")
    parser.add_argument("--scene-kmeans", action="store_true",
                        help="like --kmeans, but fit the 2-means to the SCENE alone: the "
                             "robot is removed before the cut and re-added after. The robot "
                             "sits near 0.99 while the scene occupies ~0.45-0.76, so a cut "
                             "fitted to everything splits robot-vs-scene and stops -- which "
                             "leaves 12 of 20 non-door tasks with a subgraph containing no "
                             "task object at all. Fitting to the scene takes that to 0, and "
                             "recovers 94.8%% of the human targets across 3 seeds once "
                             "`Counter` (droppable at no rollout cost) is set aside.")
    parser.add_argument("--modality", help="restrict to one graph modality (default: average all)")
    parser.add_argument("-o", "--out", help="where to write the whitelist YAML")
    args = parser.parse_args()

    per_task = {}
    for spec in args.task_threshold:
        task, _, value = spec.partition("=")
        try:
            per_task[task.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--task-threshold expects TASK=VALUE, got '{spec}'")

    snap = load_snapshot(args.snapshot)
    mode = snap.get("graph_eval_mode", "none")
    if mode not in (None, "none"):
        print(f"!!! WARNING: snapshot recorded with graph_eval_mode='{mode}' — "
              f"scores reflect an ablation run !!!")

    means = pool_means(snap, args.modality)
    stds = pooled_std(snap, args.modality)
    if not means:
        raise SystemExit("snapshot contains no scores")

    unknown = sorted(set(per_task) - set(means))
    if unknown:
        raise SystemExit(f"--task-threshold names not in the snapshot: {unknown} "
                         f"(has: {sorted(means)})")

    whitelist, warnings, used = {}, [], {}
    for task, nodes in sorted(means.items()):
        relevant = RELEVENT_NODES.get(task, set()) if RELEVENT_NODES else set()
        ranked = sorted(nodes.items(), key=lambda kv: -kv[1])
        if args.scene_kmeans:
            scene = [(n, v) for n, v in ranked if n not in ROBOT_NODES]
            scene_keep, cutoff = kmeans_split(scene) if scene else ([], 0.0)
            km_keep = set(scene_keep) | {n for n, _ in ranked if n in ROBOT_KEPT}
            tag = (f"  (scene 2-means -> cutoff {cutoff:.3f}, {len(scene_keep)} scene nodes "
                   f"+ robot = {len(km_keep)})")
        elif args.kmeans:
            km_keep, cutoff = kmeans_split(ranked)
            km_keep = set(km_keep)
            tag = f"  (2-means -> cutoff {cutoff:.3f}, {len(km_keep)} nodes)"
        else:
            km_keep = None
            cutoff = per_task.get(task, args.threshold)
            tag = f"  (cutoff {cutoff}{' — per-task' if task in per_task else ''})" if cutoff is not None else ""
        print(f"\n=== {task} ==={tag}")
        for name, mean in ranked:
            kept = name in km_keep if km_keep is not None else (cutoff is not None and mean > cutoff)
            std = stds.get(task, {}).get(name, 0.0)
            near = (cutoff is not None
                    and abs(mean - cutoff) < std)  # stage-1 score straddled the bar
            print(f"  {'KEEP' if kept else '    '} {'*' if name in relevant else ' '} "
                  f"{name:<20} {mean:.3f} ± {std:.3f}{'   <- within 1 std of the cut' if near else ''}")
        counts = "  ".join(f">{c}: {sum(1 for m in nodes.values() if m > c)}" for c in SCAN_CUTOFFS)
        print(f"  keep-count by cutoff:  {counts}")

        if km_keep is not None or cutoff is not None:
            keep = ([name for name, _ in ranked if name in km_keep] if km_keep is not None
                    else [name for name, mean in ranked if mean > cutoff])
            used[task] = round(float(cutoff), 4)
            whitelist[task] = keep
            if len(keep) == 0:
                warnings.append(f"{task}: EMPTY subgraph — the policy would be scene-blind")
            elif len(keep) == 1:
                warnings.append(f"{task}: 1-node subgraph ({keep[0]})")
            missed = sorted(relevant - set(keep))
            if relevant and missed:
                warnings.append(f"{task}: dropped RELEVENT_NODES {missed}")

    if not whitelist:
        print("\nNo --threshold/--task-threshold/--kmeans/--scene-kmeans given: "
              "tables only, nothing written.")
        return

    uncut = sorted(set(means) - set(whitelist))
    if uncut:
        warnings.append(f"no cutoff for {uncut} — absent from the whitelist, and stage 2 "
                        f"raises KeyError if it trains on them")
    for w in warnings:
        print(f"\n!!! {w}")

    doc = {
        "threshold": args.threshold,
        "selection": ("scene_kmeans" if args.scene_kmeans
                      else "kmeans" if args.kmeans else "threshold"),
        "task_thresholds": used if (per_task or args.kmeans or args.scene_kmeans) else None,
        "source": args.snapshot,
        "modality": args.modality or "all (averaged)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "tasks": whitelist,
    }
    if args.out:
        with open(args.out, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=True, default_flow_style=False)
        print(f"\nWhitelist written to {args.out}\nStage 2:\n"
              f"  python main.py method/graph_encoder=xai_gnn_vgib_stage2 "
              f"method.graph_encoder.sparsification_layer.node_whitelist={args.out}")
    else:
        print("\n" + yaml.safe_dump(doc, sort_keys=True, default_flow_style=False))
        print("(pass -o <path> to write this to a file)")


if __name__ == "__main__":
    main()
