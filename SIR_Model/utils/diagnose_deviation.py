"""Does VGIB's compression term suppress the nodes that stand out?

The truncated KL used by our VGIB has an interior optimum at

    p*_i = 1 / (1 + D_i),      D_i = mean_d( ((x_i,d - mu_G,d) / sigma_G,d)^2 )

so a node is charged in proportion to how far its features deviate from its own
graph's mean. If that term dominates, the bottleneck systematically prefers bland,
omnipresent nodes (the robot, which appears in every graph with consistent features)
and evicts distinctive ones -- which in a kitchen scene is usually the object the task
is about. That would make the mechanism structurally wrong for this domain rather than
merely mistuned.

This measures D_i exactly as `Sparsification_Module._vgib` does -- on the batched node
features that actually reach the module -- and correlates it against the scores VGIB
learned. Two predictions if the mechanism is real:

  1. learned score anti-correlates with D across node types
  2. p* = 1/(1+D) tracks the learned score, and the nodes VGIB evicts are the high-D ones

Run (CPU only, no GPU needed):
    python -m utils.diagnose_deviation manager.task_names=[ALL] \
        +snapshot=logs/robocasa_vgib/VGIB_s42_bs16/node_scores/val_epoch049.json
"""
import json
import logging
from collections import defaultdict

import hydra
import torch
from omegaconf import OmegaConf
from torch_geometric.utils import to_dense_batch

log = logging.getLogger(__name__)


def node_deviations(batch, node_mask, num_graphs):
    """D_i per node, identical to the quad term's second half inside _vgib."""
    x = batch.x.float()
    x_dense, dense_mask = to_dense_batch(x, node_mask, batch_size=num_graphs)
    valid = dense_mask.unsqueeze(-1)
    counts = dense_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    mu_g = (x_dense * valid).sum(dim=1) / counts
    var_g = ((x_dense - mu_g.unsqueeze(1)).pow(2) * valid).sum(dim=1) / counts
    std_g = var_g.clamp(min=0).sqrt() + 1e-6
    mu, std = mu_g[node_mask], std_g[node_mask]
    return ((x - mu) / std).pow(2).mean(dim=-1)


@hydra.main(config_path="../configs", config_name="main.yaml", version_base=None)
def main(cfg):
    torch.set_grad_enabled(False)
    max_batches = int(OmegaConf.select(cfg, "max_batches", default=60))
    snapshot = OmegaConf.select(cfg, "snapshot", default=None)

    manager = hydra.utils.instantiate(cfg.manager)
    manager.init_data()

    acc = defaultdict(list)          # (task, node name) -> [D_i ...]
    for bi, batch in enumerate(manager.train_dataloader):
        if bi >= max_batches:
            break
        # Same access path as Diffusion.preprocess_batch: observation -> obs_graph ->
        # {modality: Batch}. Reading it any other way risks measuring a different tensor
        # than the one the sparsifier actually sees.
        graphs = next(iter(batch["observation"]["obs_graph"].values()))
        tasks = batch.get("task_name", None)
        nm = graphs.batch
        n_graphs = int(nm.max().item()) + 1
        D = node_deviations(graphs, nm, n_graphs)
        names = graphs.node_names
        if names and isinstance(names[0], (list, tuple)):
            names = [n for sub in names for n in sub]
        for i, d in enumerate(D.tolist()):
            g = int(nm[i])
            t = tasks[g] if tasks is not None and g < len(tasks) else "ALL"
            acc[(t, names[i])].append(d)

    stats = {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}
    out = {f"{t}||{n}": {"D": d, "count": c} for (t, n), (d, c) in stats.items()}
    with open("deviations.json", "w") as f:
        json.dump(out, f, indent=1)
    log.info(f"wrote deviations.json with {len(out)} (task, node) entries")

    if snapshot:
        snap = json.load(open(hydra.utils.to_absolute_path(snapshot)))["scores"]
        mod = list(snap)[0]
        pairs = []
        for t, e in snap[mod].items():
            for n, s in e["nodes"].items():
                if (t, n) in stats:
                    pairs.append((t, n, stats[(t, n)][0], s["mean"]))
        if pairs:
            import statistics as st
            D = [p[2] for p in pairs]
            S = [p[3] for p in pairs]
            P = [1.0 / (1.0 + d) for d in D]

            def pear(a, b):
                ma, mb = st.mean(a), st.mean(b)
                num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
                return num / den if den else float("nan")

            log.info(f"n = {len(pairs)} (task, node) pairs")
            log.info(f"corr(D, learned score)      = {pear(D, S):+.3f}   "
                     f"[prediction: strongly negative]")
            log.info(f"corr(p*=1/(1+D), score)     = {pear(P, S):+.3f}   "
                     f"[prediction: strongly positive]")
            for t, n, d, s in sorted(pairs, key=lambda p: -p[2])[:12]:
                log.info(f"  highest D  {t:<22s} {n:<18s} D={d:8.2f}  p*={1/(1+d):.3f}  score={s:.3f}")


if __name__ == "__main__":
    main()
