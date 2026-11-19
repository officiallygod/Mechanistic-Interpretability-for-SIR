"""Suggest a VGIB sparsification threshold from a node-score snapshot.

Usage (from the repo root):
    python -m utils.suggest_threshold <snapshot.json> [--modality MOD]

VGIB has no k budget — `threshold` alone decides how large the extracted subgraph
is — so it cannot be guessed in advance. Train once, then read the cut off the
scores the model actually produced.

Reads one snapshot written by `NodeScoreTracker.save_json` (e.g.
node_scores/val_epoch049.json) and, per (modality, task), prints the nodes sorted
by mean score with two candidate cutoffs:
  - gap:     midpoint of the largest gap between consecutive sorted means. This is
             the one that matters: a clean gap means the scorer committed, and no
             gap means it did not and the threshold is arbitrary wherever you put
             it.
  - k-split: midpoint between the k-th and (k+1)-th node, k from
             get_num_relevant_nodes_per_task. A reference line, not a target.

RELEVENT_NODES membership is marked with `*`. That marking is diagnostic only —
nothing here feeds back into training. For a VGIB run it is the interesting
column: whether an unsupervised sparsifier rediscovered the hand-written table is
the question the whole variant exists to answer, so read the tables, not just the
suggested number.

Ends with a ready-to-paste Hydra override. It is a starting point, not ground
truth — and if the per-task cutoffs disagree wildly, one global threshold is the
wrong tool and the spread is telling you so.
"""
import argparse

from utils.node_scores import load_snapshot

try:
    from utils.generate_graph_dataset_robocasa import RELEVENT_NODES, get_num_relevant_nodes_per_task
except ImportError:
    RELEVENT_NODES = None
    get_num_relevant_nodes_per_task = None

DOOR_TASKS = {"CloseSingleDoor", "OpenSingleDoor", "CloseDoubleDoor", "OpenDoubleDoor"}


def _task_k(task):
    if get_num_relevant_nodes_per_task is not None:
        try:
            return get_num_relevant_nodes_per_task(task)
        except KeyError:
            pass
    return None


def gap_cutoff(sorted_means):
    """Midpoint of the largest gap between consecutive descending means."""
    if len(sorted_means) < 2:
        return None
    gaps = [(sorted_means[i] - sorted_means[i + 1], i) for i in range(len(sorted_means) - 1)]
    g, i = max(gaps)
    if g <= 0:
        return None
    return (sorted_means[i] + sorted_means[i + 1]) / 2


def ksplit_cutoff(sorted_means, k):
    """Midpoint between the k-th and (k+1)-th descending means."""
    if k is None or not 0 < k < len(sorted_means):
        return None
    return (sorted_means[k - 1] + sorted_means[k]) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshot", help="node_scores/*.json snapshot path")
    parser.add_argument("--modality", help="restrict to one graph modality")
    args = parser.parse_args()

    snap = load_snapshot(args.snapshot)
    mode = snap.get("graph_eval_mode", "none")
    if mode not in (None, "none"):
        print(f"!!! WARNING: snapshot recorded with graph_eval_mode='{mode}' — "
              f"scores reflect an ablation run !!!")

    scores = snap.get("scores", {})
    if args.modality:
        scores = {m: t for m, t in scores.items() if m == args.modality}
        if not scores:
            raise SystemExit(f"modality '{args.modality}' not in snapshot "
                             f"(available: {list(snap.get('scores', {}))})")

    # (task, name) -> list of per-modality means, for the cross-modality snippet
    pooled = {}
    for mod, tasks in sorted(scores.items()):
        for task, entry in sorted(tasks.items()):
            nodes = entry["nodes"]
            ranked = sorted(nodes.items(), key=lambda kv: -kv[1]["mean"])
            means = [stats["mean"] for _, stats in ranked]
            k = _task_k(task)
            relevant = RELEVENT_NODES.get(task, set()) if RELEVENT_NODES else set()

            print(f"\n=== {mod} / {task}  (graphs seen: {entry.get('graphs_seen', '?')}, k={k}) ===")
            for name, stats in ranked:
                mark = "*" if name in relevant else " "
                print(f"  {mark} {name:<20} {stats['mean']:.3f} ± {stats['std']:.3f}  (n={stats['count']})")
            if task in DOOR_TASKS:
                print("    (* = RELEVENT_NODES; door tasks count one extra scene node — the door itself)")

            gap = gap_cutoff(means)
            ksp = ksplit_cutoff(means, k)
            print(f"  candidates: gap={f'{gap:.3f}' if gap is not None else 'n/a'}"
                  f"  k-split={f'{ksp:.3f}' if ksp is not None else 'n/a'}")

            for name, stats in nodes.items():
                pooled.setdefault(task, {}).setdefault(name, []).append(stats["mean"])

    suggestions = {}
    for task, per_name in sorted(pooled.items()):
        means = sorted((sum(v) / len(v) for v in per_name.values()), reverse=True)
        cut = gap_cutoff(means)
        if cut is None:
            cut = ksplit_cutoff(means, _task_k(task))
        if cut is not None:
            suggestions[task] = round(cut, 3)

    if suggestions:
        print("\nPer-task gap cutoffs (cross-modality means):")
        for t, v in suggestions.items():
            print(f"  {t:<24} {v}")
        vals = sorted(suggestions.values())
        median = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
        spread = vals[-1] - vals[0]
        print(f"\nSuggested override (median of the above — verify the per-task tables first):")
        print(f"  'method.graph_encoder.sparsification_layer.threshold={round(median, 3)}'")
        if spread > 0.2:
            print(f"\n!!! per-task cutoffs span {spread:.3f} ({vals[0]:.3f}..{vals[-1]:.3f}) — a single "
                  f"global threshold will over-prune some tasks and under-prune others !!!")
    else:
        print("\nNo cutoff suggestions possible (too few nodes per task).")


if __name__ == "__main__":
    main()
