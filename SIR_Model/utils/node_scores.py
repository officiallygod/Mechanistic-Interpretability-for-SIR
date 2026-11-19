"""Shared per-node sparsification score tracking.

Accumulates the pre-mask sigmoid scores (`raw_probs`) stashed by
`Multi_XAI_GNN.forward` in `last_sparsification`, keyed by
(modality, task, node-name). Used by the trainer (train/val epochs), the
sequential tester and the parallel policy server, so all three splits
produce directly comparable score snapshots.
"""
import json
import logging
import os

import yaml
from collections import defaultdict
from datetime import datetime


log = logging.getLogger(__name__)


class NodeScoreTracker:
    def __init__(self):
        self.sum = defaultdict(float)     # (mod, task, name) -> sum of raw_probs
        self.sq_sum = defaultdict(float)  # (mod, task, name) -> sum of raw_probs^2
        self.cnt = defaultdict(int)       # (mod, task, name) -> steps node was present
        self.graphs = defaultdict(int)    # (mod, task) -> graphs (forward steps) seen
        self._warned_mods = set()

    def __bool__(self):
        return bool(self.cnt)

    def update(self, last_sparsification):
        for mod, info in (last_sparsification or {}).items():
            names, tasks, raw = info.get('node_names'), info.get('task_names'), info.get('raw_probs')
            if names is None or not tasks or raw is None or len(names) == 0:
                continue
            # PyG collation yields one name-list per graph; the sequential
            # tester feeds a flat list (single graph, batch size 1).
            if not isinstance(names[0], (list, tuple)):
                names = [names]
            total = sum(len(n) for n in names)
            if total != len(raw):
                if mod not in self._warned_mods:
                    log.warning(f"node_names/raw_probs length mismatch for '{mod}' "
                                f"({total} names vs {len(raw)} probs); skipping modality")
                    self._warned_mods.add(mod)
                continue
            offset = 0
            for i, names_i in enumerate(names):
                task = tasks[i] if i < len(tasks) else tasks[-1]
                self.graphs[(mod, task)] += 1
                for name in names_i:
                    v = float(raw[offset])
                    key = (mod, task, name)
                    self.sum[key] += v
                    self.sq_sum[key] += v * v
                    self.cnt[key] += 1
                    offset += 1

    def update_from_method(self, method):
        ge = getattr(method, 'graph_encoder', None)
        if hasattr(ge, 'last_sparsification'):
            self.update(ge.last_sparsification)

    def merge(self, other):
        for key, v in other.sum.items():
            self.sum[key] += v
            self.sq_sum[key] += other.sq_sum[key]
            self.cnt[key] += other.cnt[key]
        for key, v in other.graphs.items():
            self.graphs[key] += v

    def means(self):
        return {key: self.sum[key] / self.cnt[key] for key in self.sum}

    def task_node_means(self):
        """{(task, node): mean score} for this epoch, averaged across modalities.

        A node's score is a property of the node, so splitting it per modality doubles
        the series count for no extra insight at this level. The per-modality breakdown
        stays in the JSON snapshot.
        """
        agg = defaultdict(lambda: [0.0, 0])
        for (mod, task, name), s in self.sum.items():
            entry = agg[(task, name)]
            entry[0] += s
            entry[1] += self.cnt[(mod, task, name)]
        return {key: total / n for key, (total, n) in agg.items() if n}

    def save_json(self, path, metadata=None):
        """Per-modality, per-task, per-node means + stds. This is the durable record:
        it is what `utils.suggest_threshold` and `utils.select_subgraph` read, and what
        `utils.backfill_node_scores` replays when a run logged the wrong thing."""
        scores = {}
        for (mod, task, name), s in self.sum.items():
            n = self.cnt[(mod, task, name)]
            mean = s / n
            var = max(self.sq_sum[(mod, task, name)] / n - mean * mean, 0.0)
            entry = scores.setdefault(mod, {}).setdefault(
                task, {'graphs_seen': self.graphs[(mod, task)], 'nodes': {}})
            entry['nodes'][name] = {'mean': mean, 'std': var ** 0.5, 'count': n}
        out = dict(metadata or {})
        out['timestamp'] = datetime.now().isoformat(timespec='seconds')
        out['scores'] = scores
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        log.info(f"Node-score snapshot written to {path}")


def wandb_log_dict(tracker, split, epoch):
    """{metric name: value} for one epoch — plain scalars, one per (task, object).

    Deliberately *not* a `wandb.plot.line_series` custom chart. A custom chart renders
    as a static image: you cannot hover a line to read which object it is, cannot zoom,
    cannot smooth, cannot compare the same object across runs, and it never shows up in
    the run's history for the API. These are ordinary metrics instead, so an object's
    score behaves exactly like a loss curve.

    The name is `node_scores/<split>/<task>/<object>`, which is what lets a single line
    panel pull in every object of one task by regex — see utils/make_wandb_panels.py,
    which builds those panels so nobody has to click them together by hand.
    """
    out = {f"node_scores/{split}/{task}/{name}": value
           for (task, name), value in tracker.task_node_means().items()}
    if out:
        # Explicit x-axis for the node-score panels. Equal to the global step during
        # training, but `utils.backfill_node_scores` cannot reuse the global step on a
        # resumed run, so both paths agree only if the axis is its own metric.
        out["node_epoch"] = epoch
    return out


def load_snapshot(path):
    with open(path) as f:
        return json.load(f)


def load_node_whitelist(spec):
    """Resolve a stage-2 subgraph spec into {task: set(node names)}.

    `spec` is either a path to a file written by `utils.select_subgraph` or an inline
    mapping ({task: [names]}, or the same under a 'tasks' key).

    Note what this whitelist is and is not: it comes from stage 1's *own* unsupervised
    node scores, never from RELEVENT_NODES. Stage 2 distils a selection the model made
    for itself; it does not import the hand-written relevance table.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        with open(spec) as f:
            spec = yaml.safe_load(f)  # also parses the JSON variant
    else:
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(spec):
                spec = OmegaConf.to_container(spec, resolve=True)
        except ImportError:
            pass
    if not isinstance(spec, dict):
        raise ValueError(f"node whitelist must be a mapping or a path, got {type(spec).__name__}")
    tasks = spec.get('tasks', spec)
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("node whitelist has no tasks")
    out = {}
    for task, names in tasks.items():
        if isinstance(names, str) or not hasattr(names, '__iter__'):
            raise ValueError(f"whitelist entry for task '{task}' must be a list of node names")
        out[task] = set(names)
    return out
