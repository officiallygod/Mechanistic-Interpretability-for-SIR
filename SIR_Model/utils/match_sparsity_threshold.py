"""Pick the VGIB threshold whose kept-node fraction matches a target.

    python -m utils.match_sparsity_threshold <snapshot.json> [--target 0.34] [--min-nodes 2]

Prints one number: the global cutoff to pass as
`method.graph_encoder.sparsification_layer.threshold`.

Why not `utils.suggest_threshold`? That tool proposes the midpoint of the largest gap in
the score distribution, which is the right question when asking *whether the scorer
committed*. It is the wrong question when the number has to be comparable across runs: on
seed 42 its median-of-per-task-gaps was 0.847, which left OpenDrawer with an EMPTY
subgraph -- the pooled graph embedding goes to zero and the policy is scene-blind, so the
task scores 0 for a reason that has nothing to do with the method.

This picks the cutoff that keeps the same *fraction* of nodes SIR's per-task top-k keeps
(~0.34), so the two selectors are sparsity-matched and their faithfulness can be compared
directly. It reads nothing from RELEVENT_NODES, so the selection stays unsupervised --
only the operating point is matched, and the target is a single scalar taken from the
baseline's logged sparsity.

`--min-nodes` is a hard floor: a cutoff that empties or nearly empties any task's graph is
rejected however well it matches the target, because that failure mode is not a sparsity
trade-off, it is a broken evaluation.
"""
import argparse

import numpy as np

from utils.node_scores import load_snapshot
from utils.select_subgraph import pool_means


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("snapshot")
    ap.add_argument("--target", type=float, default=0.34,
                    help="node fraction to keep (default 0.34 = SIR's logged sparsity)")
    ap.add_argument("--min-nodes", type=int, default=2,
                    help="reject any cutoff leaving fewer than this on some task")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    means = pool_means(load_snapshot(a.snapshot))
    if not means:
        raise SystemExit("snapshot contains no scores")

    best, best_err = None, None
    for thr in np.arange(0.05, 0.99, 0.005):
        keeps = {t: sum(1 for v in n.values() if v > thr) for t, n in means.items()}
        if min(keeps.values()) < a.min_nodes:
            continue
        frac = sum(keeps.values()) / sum(len(n) for n in means.values())
        err = abs(frac - a.target)
        if best_err is None or err < best_err:
            best, best_err, best_frac, best_min = float(thr), err, frac, min(keeps.values())

    if best is None:
        raise SystemExit(f"no cutoff keeps >= {a.min_nodes} nodes on every task; "
                         f"the scorer did not separate this run's nodes")
    if a.verbose:
        print(f"# threshold {best:.3f} keeps {best_frac:.3f} of nodes "
              f"(target {a.target}), min {best_min} nodes/task", flush=True)
    print(f"{best:.3f}")


if __name__ == "__main__":
    main()
