"""Compare per-node sparsification score snapshots for subgraph consistency.

Usage (from the repo root):
    python -m utils.compare_node_scores <eval_run_dir> \
        [--train-run-dir DIR | --wandb-run ENTITY/PROJECT/RUN_ID] \
        [--epoch-tag final] [--csv out.csv]

Loads the per-round eval snapshots (node_scores/eval_{tag}_round*.json) written
by RoboCasa_Manager.test_method and reports, per (modality, task):
  - cross-round consistency: mean pairwise Spearman over shared node names,
    top-k Jaccard, per-node coefficient of variation
  - eval vs train/val: Spearman + top-k Jaccard, plus names present only on
    one side (live segmentation vs dataset vocabulary drift)
  - precision@k of the merged eval top-k against RELEVENT_NODES ground truth

Train/val snapshots come from --train-run-dir (default: the eval run dir, then
trainer.model_path from its .hydra/config.yaml), or from the wandb API via
--wandb-run for models trained before snapshots were written to disk.
"""
import argparse
import csv
import glob
import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr

try:
    from utils.generate_graph_dataset_robocasa import RELEVENT_NODES, get_num_relevant_nodes_per_task
except ImportError:
    RELEVENT_NODES = None
    get_num_relevant_nodes_per_task = None

DEFAULT_K = 5
DOOR_TASKS = {"CloseSingleDoor", "OpenSingleDoor", "CloseDoubleDoor", "OpenDoubleDoor"}


def _task_k(task):
    if get_num_relevant_nodes_per_task is not None:
        try:
            return get_num_relevant_nodes_per_task(task)
        except KeyError:
            pass
    return DEFAULT_K


def _snapshot_means(snapshot):
    """Flatten a snapshot JSON into {mod: {task: {name: mean}}}."""
    out = {}
    for mod, tasks in snapshot.get('scores', {}).items():
        for task, entry in tasks.items():
            out.setdefault(mod, {})[task] = {
                name: stats['mean'] for name, stats in entry['nodes'].items()}
    return out


def load_eval_rounds(run_dir, tag):
    paths = sorted(glob.glob(os.path.join(run_dir, 'node_scores', f'eval_{tag}_round*.json')))
    rounds = []
    for p in paths:
        with open(p) as f:
            snap = json.load(f)
        mode = snap.get('graph_eval_mode', 'none')
        if mode != 'none':
            print(f"!!! WARNING: {os.path.basename(p)} was recorded with "
                  f"graph_eval_mode='{mode}' — scores reflect an ablation run !!!")
        rounds.append(_snapshot_means(snap))
    return paths, rounds


def load_train_val_json(train_dir):
    """Latest-epoch train/val snapshot means from <dir>/node_scores/."""
    out = {}
    for split in ('train', 'val'):
        paths = sorted(glob.glob(os.path.join(train_dir, 'node_scores', f'{split}_epoch*.json')))
        if paths:
            with open(paths[-1]) as f:
                out[split] = _snapshot_means(json.load(f))
            print(f"[{split}] using {paths[-1]}")
    return out


def load_train_val_wandb(run_path):
    """Fetch last-logged node_scores/train|val/... values from the wandb API."""
    import wandb
    run = wandb.Api().run(run_path)
    out = {}
    for key, v in dict(run.summary).items():
        parts = key.split('/')
        if len(parts) != 5 or parts[0] != 'node_scores' or parts[1] not in ('train', 'val'):
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        _, split, mod, task, name = parts
        out.setdefault(split, {}).setdefault(mod, {}).setdefault(task, {})[name] = val
    for split in out:
        print(f"[{split}] fetched from wandb run {run_path}")
    return out


def resolve_train_dir(eval_run_dir, train_run_dir):
    if train_run_dir:
        return train_run_dir
    if glob.glob(os.path.join(eval_run_dir, 'node_scores', '*_epoch*.json')):
        return eval_run_dir
    hydra_cfg = os.path.join(eval_run_dir, '.hydra', 'config.yaml')
    if os.path.exists(hydra_cfg):
        try:
            from omegaconf import OmegaConf
            model_path = OmegaConf.select(OmegaConf.load(hydra_cfg), 'trainer.model_path')
            if model_path and os.path.isdir(str(model_path)):
                print(f"[train/val] derived from trainer.model_path: {model_path}")
                return str(model_path)
        except Exception as e:
            print(f"[train/val] could not derive model_path from {hydra_cfg}: {e}")
    return None


def spearman(a, b):
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return float('nan'), len(shared)
    rho = spearmanr([a[n] for n in shared], [b[n] for n in shared]).correlation
    return rho, len(shared)


def topk(scores, k):
    return set(sorted(scores, key=scores.get, reverse=True)[:k])


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else float('nan')


def analyze(rounds, reference, mod, task, csv_rows):
    """Print + collect consistency metrics for one (modality, task)."""
    per_round = [r[mod][task] for r in rounds if task in r.get(mod, {})]
    if not per_round:
        return
    k = _task_k(task)
    print(f"\n=== {mod} / {task}  (rounds: {len(per_round)}, k={k}) ===")

    def emit(metric, value):
        csv_rows.append({'mod': mod, 'task': task, 'metric': metric, 'value': value})

    emit('n_rounds', len(per_round))
    emit('k', k)

    # ---- cross-round consistency ----
    if len(per_round) >= 2:
        rhos, jacs = [], []
        for a, b in itertools.combinations(per_round, 2):
            rho, _ = spearman(a, b)
            if rho == rho:
                rhos.append(rho)
            jacs.append(jaccard(topk(a, k), topk(b, k)))
        rho_m = float(np.mean(rhos)) if rhos else float('nan')
        jac_m = float(np.nanmean(jacs)) if jacs else float('nan')
        print(f"cross-round:  spearman={rho_m:.3f}  top-{k} jaccard={jac_m:.3f}")
        emit('cross_round_spearman', rho_m)
        emit('cross_round_topk_jaccard', jac_m)

        union = set().union(*per_round)
        cvs = {}
        for name in union:
            vals = [r[name] for r in per_round if name in r]
            if len(vals) >= 2 and np.mean(vals) > 0:
                cvs[name] = float(np.std(vals) / np.mean(vals))
        for name, cv in sorted(cvs.items(), key=lambda kv: -kv[1]):
            emit(f'node_cv/{name}', cv)
        unstable = sorted(cvs.items(), key=lambda kv: -kv[1])[:5]
        if unstable:
            print("most unstable nodes (cv): "
                  + ", ".join(f"{n} ({cv:.3f})" for n, cv in unstable))

    # merged eval means (rounds are equal-sized by design)
    merged = {}
    for name in set().union(*per_round):
        merged[name] = float(np.mean([r[name] for r in per_round if name in r]))

    # ---- eval vs train/val ----
    for split, ref in (reference or {}).items():
        ref_mt = ref.get(mod, {}).get(task)
        if not ref_mt:
            continue
        rho, n_shared = spearman(merged, ref_mt)
        jac = jaccard(topk(merged, k), topk(ref_mt, k))
        print(f"eval vs {split}: spearman={rho:.3f}  top-{k} jaccard={jac:.3f}  "
              f"(shared names: {n_shared})")
        emit(f'eval_vs_{split}_spearman', rho)
        emit(f'eval_vs_{split}_topk_jaccard', jac)
        eval_only = sorted(set(merged) - set(ref_mt))
        ref_only = sorted(set(ref_mt) - set(merged))
        if eval_only:
            print(f"  names only in eval:  {eval_only}")
        if ref_only:
            print(f"  names only in {split}: {ref_only}")

    # ---- merged eval top-k vs ground truth ----
    if RELEVENT_NODES is not None and task in RELEVENT_NODES:
        relevant = RELEVENT_NODES[task]
        sel = topk(merged, k)
        prec = len(sel & relevant) / len(sel) if sel else float('nan')
        note = " (door task: goal-dependent door node not counted)" if task in DOOR_TASKS else ""
        print(f"precision@{k} vs RELEVENT_NODES: {prec:.3f}{note}")
        emit('precision_at_k', prec)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('eval_run_dir', help='hydra run dir containing node_scores/eval_*.json')
    parser.add_argument('--train-run-dir', default=None,
                        help='run dir with node_scores/{train,val}_epoch*.json '
                             '(default: eval dir, then trainer.model_path from .hydra config)')
    parser.add_argument('--wandb-run', default=None,
                        help='ENTITY/PROJECT/RUN_ID — fetch train/val scores from the wandb API '
                             '(for models trained before JSON snapshots existed)')
    parser.add_argument('--epoch-tag', default='final',
                        help="'final' or 'epoch_005' — which eval snapshots to load")
    parser.add_argument('--csv', default=None, help='write long-form metrics table to this path')
    args = parser.parse_args()

    paths, rounds = load_eval_rounds(args.eval_run_dir, args.epoch_tag)
    if not rounds:
        raise SystemExit(f"no eval snapshots matching 'eval_{args.epoch_tag}_round*.json' "
                         f"under {os.path.join(args.eval_run_dir, 'node_scores')}")
    print(f"loaded {len(rounds)} eval round snapshot(s):")
    for p in paths:
        print(f"  {p}")

    if args.wandb_run:
        reference = load_train_val_wandb(args.wandb_run)
    else:
        train_dir = resolve_train_dir(args.eval_run_dir, args.train_run_dir)
        reference = load_train_val_json(train_dir) if train_dir else {}
    if not reference:
        print("no train/val reference found — reporting eval cross-round consistency only")

    csv_rows = []
    mod_tasks = sorted({(mod, task) for r in rounds
                        for mod, tasks in r.items() for task in tasks})
    for mod, task in mod_tasks:
        analyze(rounds, reference, mod, task, csv_rows)

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['mod', 'task', 'metric', 'value'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote {len(csv_rows)} rows to {args.csv}")


if __name__ == '__main__':
    main()
