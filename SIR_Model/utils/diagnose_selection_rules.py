"""Score candidate selection rules against the human list, on snapshots we already have.

Why
---
The ranking VGIB learns is largely right and the CUT is what is wrong. Median rank of a
human-labelled target inside VGIB's own ordering is 5, and 33/44 targets sit in the top 8
-- but `select_subgraph --kmeans` cuts at 2-3 nodes, because a 2-means fit on a
distribution where the robot sits at ~0.99 and the whole scene at 0.45-0.76 splits
robot-vs-everything and stops.

So the rule, not the scorer, is the thing to fix. This scores candidate rules offline so
only a survivor costs GPU time.

Robot nodes are excluded from the SUBGRAPH here, not merely from the cut. An earlier
version force-kept them and was falsified by its own control: because the robot IS the
top of the ranking, holding it out and cutting the rest is arithmetically just a lower
threshold, and it scored slightly worse than plain top-k at matched size by spending a
slot on `OmronMobileBase`. The robot is trivially relevant to every task and so carries
no information about which task this is; an explanation should be the scene.

Reading the output
------------------
Precision is measured against RELEVENT_NODES, which we have evidence is NOT optimal -- it
omits `drawer_obj` on CloseDrawer and `cookware` on TurnOffStove, both of which VGIB
found and both of which coincided with a rollout gain. So precision here is "agreement
with the human list", not correctness, and a rule that loses a little precision by adding
plausible manipulanda is not necessarily worse. Recall is the safer signal.

Door tasks are excluded throughout: SIR resolves the door from each episode's language,
so its static entry is robot-only and a per-task list cannot be scored against it.

Usage:
    python -m utils.diagnose_selection_rules          # val snapshots (the honest basis)
    python -m utils.diagnose_selection_rules train    # train snapshots, for contrast only
"""
import glob
import statistics
import sys

from utils.node_scores import load_snapshot
from utils.select_subgraph import kmeans_split, pool_means

try:
    from utils.generate_graph_dataset_robocasa import RELEVENT_NODES
except ImportError:
    RELEVENT_NODES = None

ROBOT = {"PandaMobile", "PandaGripper", "OmronMobileBase"}
DOORS = {"OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor"}


def scene_ranked(means):
    """The scene, best first, with the robot removed entirely."""
    return sorted(((n, v) for n, v in means.items() if n not in ROBOT),
                  key=lambda kv: -kv[1])


def rule_all_kmeans(means):
    """What we ship today: one 2-means cut over everything, robot included."""
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    kept, _ = kmeans_split(ranked)
    return {n for n in kept if n not in ROBOT}


def rule_scene_kmeans(means):
    """Same cut, but fitted to the scene alone so the robot cannot define the split."""
    ranked = scene_ranked(means)
    if not ranked:
        return set()
    kept, _ = kmeans_split(ranked)
    return set(kept)


def rule_scene_topk(k):
    def f(means):
        return {n for n, _ in scene_ranked(means)[:k]}
    f.__name__ = f"scene_top{k}"
    return f


def rule_scene_gap(means):
    """Cut at the largest consecutive drop in the scene's scores -- the other obvious
    unsupervised rule, and one that does not assume two clusters exist."""
    ranked = scene_ranked(means)
    if len(ranked) < 2:
        return {n for n, _ in ranked}
    vals = [v for _, v in ranked]
    gaps = [(vals[i] - vals[i + 1], i + 1) for i in range(len(vals) - 1)]
    _, cut = max(gaps)
    return {n for n, _ in ranked[:cut]}


def rule_scene_relgap(means):
    """Largest gap measured relative to the spread below it -- less biased toward cutting
    after node 1, which plain max-gap does whenever one object stands out."""
    ranked = scene_ranked(means)
    if len(ranked) < 3:
        return {n for n, _ in ranked}
    vals = [v for _, v in ranked]
    best, cut = -1.0, 1
    for i in range(1, len(vals) - 1):
        spread = vals[i] - vals[-1]
        score = (vals[i - 1] - vals[i]) / (spread + 1e-9)
        if score > best:
            best, cut = score, i
    return {n for n, _ in ranked[:cut]}


def targets_for(task):
    if RELEVENT_NODES is None:
        return set()
    return {n for n in RELEVENT_NODES.get(task, ()) if n not in ROBOT}


def main():
    rules = [
        ("all_kmeans (current)", rule_all_kmeans),
        ("scene_kmeans", rule_scene_kmeans),
        ("scene_maxgap", rule_scene_gap),
        ("scene_relgap", rule_scene_relgap),
    ] + [(f"scene_top{k}", rule_scene_topk(k)) for k in (2, 3, 4, 5, 6, 8)]

    # val, not train. The two disagree sharply -- on seed 42 scene_kmeans scores 100%
    # excl-Counter recall on the train snapshot and 65.6% on val -- because train pools
    # ~14k graphs per task against val's ~826, so train means are far better resolved.
    # But the whitelist is applied to scenes the model has not seen, and the existing
    # artifacts were all cut from val, so val is the honest basis and the comparable one.
    # Train numbers flatter every rule and must not be quoted as the expected result.
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    snaps = {}
    for seed in ("42", "43", "44"):
        f = sorted(glob.glob(
            f"logs/robocasa_vgib/VGIB_s{seed}_bs16/node_scores/{split}_epoch*.json"))
        if f:
            snaps[seed] = pool_means(load_snapshot(f[-1]))
    if not snaps:
        sys.exit("no snapshots found")

    hdr = (f"{'rule':<22}{'size':>6}{'recall':>9}{'prec':>8}{'F1':>8}"
           f"{'tasks w/ 0 targets':>20}")
    print(f"\n{len(snaps)} seeds, {len([t for t in next(iter(snaps.values())) if t not in DOORS])} "
          f"non-door tasks, robot excluded from every subgraph\n")
    print(hdr)
    print("-" * len(hdr))
    for name, rule in rules:
        recs, precs, sizes, empties = [], [], [], []
        for seed, means in snaps.items():
            hit = tot_t = tot_k = 0
            miss = 0
            for task, m in means.items():
                if task in DOORS:
                    continue
                kept = rule(m)
                tg = targets_for(task)
                hit += len(tg & kept)
                tot_t += len(tg)
                tot_k += len(kept)
                sizes.append(len(kept))
                if tg and not (tg & kept):
                    miss += 1
            recs.append(hit / tot_t if tot_t else 0)
            precs.append(hit / tot_k if tot_k else 0)
            empties.append(miss)
        r, p = statistics.mean(recs), statistics.mean(precs)
        f1 = 2 * r * p / (r + p) if (r + p) else 0
        print(f"{name:<22}{statistics.mean(sizes):>6.1f}{100*r:>8.0f}%{100*p:>7.0f}%"
              f"{100*f1:>7.0f}%{statistics.mean(empties):>17.1f}/20")


if __name__ == "__main__":
    main()
