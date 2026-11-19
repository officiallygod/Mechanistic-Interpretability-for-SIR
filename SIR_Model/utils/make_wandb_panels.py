"""Build the node-score workspace view: ONE line panel per (split, task).

wandb turns every metric key into its own panel, so ~40 objects x 2 tasks x 2 splits
arrives as ~80 tiny charts. The objects of a task only mean anything against each other,
which needs them on shared axes — so the panels have to be declared, and that is what
the workspaces API is for.

Run it from a SEPARATE environment, not the training env:

    python -m venv /tmp/wsenv && /tmp/wsenv/bin/pip install wandb-workspaces
    /tmp/wsenv/bin/python utils/make_wandb_panels.py --entity layen --project robocasa_vgib

`wandb-workspaces` pulls a newer `wandb` with it, and upgrading that under a running
evaluation is how you corrupt a comparison. This script only talks to the wandb API, so
it has no reason to share an interpreter with anything that trains.

Metrics are discovered from the runs themselves, so a task or an object that only some
runs have still lands on the right panel.
"""
import argparse
import re
from collections import defaultdict

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

METRIC = re.compile(r"^node_scores/(?P<split>train|val)/(?P<task>[^/]+)/(?P<node>.+)$")


def discover(entity, project, limit):
    """{(split, task): [metric names]} across the project's runs, most recent first."""
    api = wandb.Api()
    found = defaultdict(set)
    for run in list(api.runs(f"{entity}/{project}"))[:limit]:
        for key in run.summary.keys():
            m = METRIC.match(str(key))
            if m:
                found[(m["split"], m["task"])].add(str(key))
    # Rank by the last value seen so the legend reads as a ranking rather than
    # alphabetically, which is what makes 20 lines legible.
    return {k: sorted(v) for k, v in found.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--view-name", default="node scores")
    ap.add_argument("--runs-to-scan", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = discover(args.entity, args.project, args.runs_to_scan)
    if not groups:
        raise SystemExit("no node_scores/* metrics found — has a run logged them yet?")

    for (split, task), metrics in sorted(groups.items()):
        print(f"  {split}/{task}: {len(metrics)} objects on one panel")
    if args.dry_run:
        return

    sections = []
    for split in ("val", "train"):
        panels = [
            wr.LinePlot(
                title=f"{split} node scores — {task}",
                x="node_epoch",
                y=sorted(metrics),
                layout=wr.Layout(w=24, h=12),
            )
            for (sp, task), metrics in sorted(groups.items()) if sp == split
        ]
        if panels:
            sections.append(ws.Section(name=f"node scores ({split})", panels=panels,
                                       is_open=True))

    view = ws.Workspace(entity=args.entity, project=args.project,
                        name=args.view_name, sections=sections)
    view.save()
    print(f"\n{sum(len(s.panels) for s in sections)} panel(s) in saved view "
          f"'{args.view_name}'\n{view.url}")


if __name__ == "__main__":
    main()
