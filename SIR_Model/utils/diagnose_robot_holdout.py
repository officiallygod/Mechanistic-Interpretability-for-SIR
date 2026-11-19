"""Does holding the robot out of the k-means cut recover the task objects?

Why ask
-------
The 1D 2-means cut in `select_subgraph.kmeans_split` runs over every node's score. The
robot dominates that distribution -- `PandaMobile` sits near 0.99 while the entire rest
of the scene occupies roughly 0.45-0.76 -- so on 24 tasks the top-ranked node is a robot
node 23 times and BOTH top slots are robot 15 times. The split therefore tends to peel
off the robot and stop, which is mechanically why 5 of 24 discovered subgraphs contain
nothing but robot nodes.

The robot is never the interesting part of an explanation: it is trivially relevant to
every task, so it carries no information about which task this is. This script asks what
the cut finds when the robot is set aside first -- kept unconditionally, and excluded
from the statistics that decide where the bar falls.

This is a pure re-read of existing snapshots. It trains nothing and changes no run.

Reading the output
------------------
`target recall` is over the NON-ROBOT members of RELEVENT_NODES, since the robot is kept
by construction in the holdout arm and scoring it would flatter both arms equally.

The four door tasks are marked `*` and should be read separately: SIR resolves the door
from each episode's language via `is_node_relevant_for_task`, so its effective list is
robot + the door named in THIS instruction. A static per-task list cannot express that,
and the static RELEVENT_NODES entry those tasks compare against is robot-only -- so their
recall is vacuous, not good.

Usage:
    python -m utils.diagnose_robot_holdout            # seeds 42 43 44
    python -m utils.diagnose_robot_holdout --seeds 42 --verbose
"""
import argparse
import glob
import os

from utils.node_scores import load_snapshot
from utils.select_subgraph import kmeans_split, pool_means

try:
    from utils.generate_graph_dataset_robocasa import RELEVENT_NODES
except ImportError:
    RELEVENT_NODES = None

ROBOT = {"PandaMobile", "PandaGripper", "OmronMobileBase"}
DOORS = {"OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor"}


def select_current(means):
    """What select_subgraph --kmeans does today: one cut over everything."""
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    kept, _ = kmeans_split(ranked)
    return set(kept)


def select_holdout(means):
    """Robot kept unconditionally; the cut is decided by the rest of the scene alone."""
    robot = {n for n in means if n in ROBOT}
    rest = sorted(((n, v) for n, v in means.items() if n not in ROBOT),
                  key=lambda kv: -kv[1])
    if not rest:
        return set(robot)
    kept, _ = kmeans_split(rest)
    return robot | set(kept)


def targets_for(task):
    """Non-robot nodes the human list names for this task."""
    if RELEVENT_NODES is None:
        return set()
    return {n for n in RELEVENT_NODES.get(task, ()) if n not in ROBOT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--root", default="logs/robocasa_vgib")
    ap.add_argument("--verbose", action="store_true",
                    help="print the kept set for every task, not just the ones that move")
    args = ap.parse_args()

    for seed in args.seeds:
        d = os.path.join(args.root, f"VGIB_s{seed}_bs16", "node_scores")
        snaps = sorted(glob.glob(os.path.join(d, "train_epoch*.json")))
        if not snaps:
            print(f"seed {seed}: no snapshots under {d}")
            continue
        means = pool_means(load_snapshot(snaps[-1]))

        print(f"\n{'='*100}\nseed {seed}   ({os.path.basename(snaps[-1])})\n{'='*100}")
        hdr = f"{'task':<24}{'n cur':>6}{'n hold':>7}  {'recall cur':>10}{'recall hold':>12}   newly kept"
        print(hdr)
        print("-" * len(hdr))

        tot = {"cur": [0, 0], "hold": [0, 0]}      # [hits, targets]
        sizes = {"cur": 0, "hold": 0}
        robot_only = {"cur": 0, "hold": 0}
        moved = 0

        for task in sorted(means):
            cur, hold = select_current(means[task]), select_holdout(means[task])
            tgt = targets_for(task)
            hc, hh = len(tgt & cur), len(tgt & hold)
            tot["cur"][0] += hc; tot["cur"][1] += len(tgt)
            tot["hold"][0] += hh; tot["hold"][1] += len(tgt)
            sizes["cur"] += len(cur); sizes["hold"] += len(hold)
            if cur and all(n in ROBOT for n in cur):
                robot_only["cur"] += 1
            if hold and all(n in ROBOT for n in hold):
                robot_only["hold"] += 1

            gained = sorted(hold - cur)
            if gained:
                moved += 1
            if args.verbose or gained:
                rc = f"{hc}/{len(tgt)}" if tgt else "-"
                rh = f"{hh}/{len(tgt)}" if tgt else "-"
                mark = " *" if task in DOORS else "  "
                star = " <-- recovers a target" if hh > hc else ""
                print(f"{task:<22}{mark}{len(cur):>6}{len(hold):>7}  {rc:>10}{rh:>12}   "
                      f"{','.join(gained)[:44]}{star}")

        n = len(means)
        print("-" * len(hdr))
        for arm in ("cur", "hold"):
            hits, targ = tot[arm]
            label = "current (one cut over all)" if arm == "cur" else "robot held out of cut"
            print(f"{label:<34} mean size {sizes[arm]/n:4.1f}   "
                  f"target recall {hits}/{targ} = {100*hits/targ if targ else 0:5.1f}%   "
                  f"robot-only subgraphs {robot_only[arm]}/{n}")
        print(f"tasks whose subgraph changes at all: {moved}/{n}")

        nd = [t for t in means if t not in DOORS]
        for arm, sel in (("cur", select_current), ("hold", select_holdout)):
            h = sum(len(targets_for(t) & sel(means[t])) for t in nd)
            g = sum(len(targets_for(t)) for t in nd)
            label = "current" if arm == "cur" else "holdout"
            print(f"  non-door only ({len(nd)} tasks)  {label:<8} target recall "
                  f"{h}/{g} = {100*h/g if g else 0:5.1f}%")


if __name__ == "__main__":
    main()
