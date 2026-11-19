"""Push per-object node scores from on-disk snapshots into an existing wandb run.

Usage (from the repo root):
    python -m utils.backfill_node_scores <run_dir> --wandb-run <entity>/<project>/<id>

Every run writes `node_scores/{train,val}_epoch*.json` regardless of what it sent to
wandb, so a run that logged the wrong thing — or logged nothing — can be given the right
metrics after the fact instead of being retrained.

The scores are logged against their own `node_epoch` x-axis rather than wandb's global
step. Resuming a finished run continues from its last step, so replaying at the original
step numbers would be silently dropped as out-of-order; a dedicated step metric sidesteps
that and still plots against epoch.
"""
import argparse
import glob
import json
import os
import re


def snapshots(run_dir, split):
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "node_scores", f"{split}_epoch*.json"))):
        m = re.search(r"epoch(\d+)\.json$", path)
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def scalars_from(path, split):
    """{metric: value} for one snapshot, pooled over modalities exactly as training does."""
    blob = json.load(open(path))["scores"]
    agg = {}
    for _mod, tasks in blob.items():
        for task, entry in tasks.items():
            for name, st in entry["nodes"].items():
                total, count = agg.get((task, name), (0.0, 0))
                agg[(task, name)] = (total + st["mean"] * st["count"], count + st["count"])
    return {f"node_scores/{split}/{task}/{name}": total / n
            for (task, name), (total, n) in agg.items() if n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--wandb-run", required=True, help="entity/project/run_id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entity, project, run_id = args.wandb_run.split("/")
    found = {s: snapshots(args.run_dir, s) for s in ("train", "val")}
    n = sum(len(v) for v in found.values())
    if not n:
        raise SystemExit(f"no node_scores snapshots under {args.run_dir}")
    print(f"{args.run_dir}: " + ", ".join(f"{len(v)} {s}" for s, v in found.items()))

    if args.dry_run:
        for split, snaps in found.items():
            if snaps:
                sc = scalars_from(snaps[-1][1], split)
                print(f"  {split} epoch {snaps[-1][0]}: {len(sc)} metrics, e.g. "
                      f"{dict(list(sorted(sc.items()))[:2])}")
        return

    import wandb
    run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
    # Without this the scores hang off wandb's global step, which a resumed run has
    # already advanced past.
    wandb.define_metric("node_epoch")
    wandb.define_metric("node_scores/*", step_metric="node_epoch")

    epochs = sorted({e for snaps in found.values() for e, _ in snaps})
    for ep in epochs:
        payload = {"node_epoch": ep}
        for split, snaps in found.items():
            for e, path in snaps:
                if e == ep:
                    payload.update(scalars_from(path, split))
        wandb.log(payload)
    print(f"backfilled {len(epochs)} epochs into {args.wandb_run}")
    run.finish()


if __name__ == "__main__":
    main()
