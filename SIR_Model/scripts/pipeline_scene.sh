#!/bin/bash
# Chains behind the udw run: decides whether udw's eval is worth 8h, then trains stage 2
# on the scene_kmeans whitelist.
#
# Why the udw eval is conditional
#
#   `uniform_dist_weight=1e-1` was queued to widen a score band too compressed to threshold
#   (median non-robot spread 0.377, ~20 nodes inside it). Whether it worked is answerable
#   for FREE from its node_scores snapshot -- the 8h rollout eval measures something else
#   entirely, because the run evals in `soft` mode where nothing is pruned. So: read the
#   selection quality off the snapshot, and only pay for the eval if the selection moved.
#
#   Seed-42 baseline under the shipping rule (all_kmeans, val_epoch049): recall 25.0%,
#   and 12 of 20 non-door tasks with a subgraph containing NO task object.
#
# Why scene_kmeans
#
#   The same 2-means, fitted to the scene with the robot removed rather than to everything.
#   The robot sits near 0.99 while the scene occupies ~0.45-0.76, so a cut fitted to all
#   nodes splits robot-vs-scene and stops. On val across 3 seeds that leaves 11.0/20 tasks
#   with no task object; scene_kmeans takes it to 4.0/20 at recall 27% -> 56%.
#
#   Read those on VAL. The train snapshot pools ~14k graphs per task against val's ~826 and
#   flatters every rule -- scene_kmeans scores 56% on val and 70% on train. The whitelist is
#   applied to unseen scenes, so val is the honest basis.
set -u
cd /home/mattes/repos/sir
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib
UDW=VGIB_s42_bs16_udw1e-1
WL="$ROOT/VGIB_s42_bs16/subgraph_scene_kmeans.yaml"
NAME=VGIB_s42_bs16_stage2_scenekm

[ -f "$WL" ] || { echo "=== ABORT: no whitelist at $WL ==="; exit 1; }

# --- wait for udw training ------------------------------------------------------------
echo "=== [$(date '+%F %T')] waiting for $UDW training ==="
until grep -q "Training done." "$ROOT/$UDW/main.log" 2>/dev/null; do
  if ! tmux has-session -t sir_next 2>/dev/null; then
    sleep 30
    grep -q "Training done." "$ROOT/$UDW/main.log" 2>/dev/null && break
    echo "=== [$(date '+%F %T')] ABORT: sir_next gone before $UDW finished training ==="
    exit 1
  fi
  sleep 60
done
echo "=== [$(date '+%F %T')] $UDW training done ==="

# --- did the histogram loss actually improve the SELECTION? ----------------------------
python3 - "$ROOT/$UDW" > /tmp/udw_verdict.txt 2>&1 <<'PY'
import glob, sys, os
sys.path.insert(0, "/home/mattes/repos/sir")
from utils.node_scores import load_snapshot
from utils.select_subgraph import pool_means
from utils.diagnose_selection_rules import rule_all_kmeans, targets_for, DOORS, ROBOT

snaps = sorted(glob.glob(os.path.join(sys.argv[1], "node_scores", "val_epoch*.json")))
if not snaps:
    print("NO_SNAPSHOT"); raise SystemExit
means = pool_means(load_snapshot(snaps[-1]))
hit = tot = empty = n = 0
bands = []
for t, m in means.items():
    if t in DOORS:
        continue
    tg, k = targets_for(t), rule_all_kmeans(m)
    hit += len(tg & k); tot += len(tg); n += 1
    if tg and not (tg & k):
        empty += 1
    nr = [v for name, v in m.items() if name not in ROBOT]
    bands.append(max(nr) - min(nr))
bands.sort()
recall = 100.0 * hit / tot if tot else 0.0
print(f"snapshot {os.path.basename(snaps[-1])}")
print(f"recall {recall:.1f}%  (baseline 25.0%)")
print(f"empty {empty}/{n}  (baseline 12/20)")
print(f"median non-robot band {bands[len(bands)//2]:.3f}  (baseline 0.371, non-door)")
# Worth 8h only if the cut genuinely improved: 3 fewer blind tasks, or +10pp recall.
print("EVAL" if (empty <= 9 or recall >= 35.0) else "SKIP")
PY
cat /tmp/udw_verdict.txt
VERDICT=$(tail -1 /tmp/udw_verdict.txt)

if [ "$VERDICT" = "EVAL" ]; then
  echo "=== [$(date '+%F %T')] udw improved the selection -- letting its eval run ==="
  until [ -n "$(ls "$ROOT/eval/${UDW}_final"/eval_per_scene_*.json 2>/dev/null)" ]; do
    tmux has-session -t sir_next 2>/dev/null || { echo "=== sir_next gone mid-eval ==="; break; }
    sleep 120
  done
else
  echo "=== [$(date '+%F %T')] udw did NOT improve the selection ($VERDICT) -- skipping its 8h eval ==="
  tmux kill-session -t sir_next 2>/dev/null
  sleep 5
  # The eval spawns workers that outlive the pane. Match on the run dir, never on a script
  # name: the tmux SERVER's cmdline is permanently the command that first started it, so a
  # name-based guard once matched its own host and idled the GPU for 4h54m.
  pkill -f "${UDW}_final" 2>/dev/null
  sleep 10
  rm -rf "$ROOT/eval/${UDW}_final"
  echo "=== [$(date '+%F %T')] udw eval cancelled; its snapshot verdict is the deliverable ==="
fi

# --- GPU handoff ----------------------------------------------------------------------
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  echo "=== [$(date '+%F %T')] waiting: GPU busy ==="
  sleep 60
done

# --- stage 2 on the scene_kmeans whitelist --------------------------------------------
# Same warm start and same seed as the shipped stage-2 run, so the whitelist is the single
# variable against VGIB_s42_bs16_stage2_lr1e-4 (14.56). pos_weight is re-derived from THIS
# whitelist: 144 positives / 260 negatives = 1.81, against 3.69 for the smaller kmeans
# subgraph. Too high and the BCE flattens the scorer into an input-independent constant;
# sparsification/{train,val}_score_std_* is the canary and collapses toward 0 an epoch or
# two before the means visibly move.
echo "=== [$(date '+%F %T')] START train $NAME ==="
if [ -f "$ROOT/$NAME/action_generator_final.pth" ]; then
  echo "=== [$(date '+%F %T')] SKIP train $NAME (exists) ==="
else
  python main.py manager.task_names=[ALL] manager.batch_size=16 \
    trainer.train_bool=true trainer.test_bool=false trainer.seed=42 \
    method/graph_encoder=xai_gnn_vgib_stage2 \
    method.graph_encoder.sparsification_layer.node_whitelist="$WL" \
    method.graph_encoder.sparsification_layer.whitelist_pos_weight=1.81 \
    trainer.init_from="$ROOT/VGIB_s42_bs16" \
    wandb.run_name="$NAME" \
    hydra.run.dir="$ROOT/$NAME"
  rc=$?
  echo "=== [$(date '+%F %T')] END train $NAME (exit $rc) ==="
fi

# --- eval -----------------------------------------------------------------------------
# Stage 2 selects with its own scalar probs > 0.5 -- the BCE drives scores to the extremes,
# so no per-task threshold is passed. If stage 2 still needs one, it did not do its job.
if [ -n "$(ls "$ROOT/eval/${NAME}_final"/eval_per_scene_*.json 2>/dev/null)" ]; then
  echo "=== [$(date '+%F %T')] SKIP eval ${NAME}_final (results present) ==="
elif [ ! -f "$ROOT/$NAME/action_generator_final.pth" ]; then
  echo "=== [$(date '+%F %T')] ABORT eval: $NAME has no _final checkpoint ==="
  exit 1
else
  rm -rf "$ROOT/eval/${NAME}_final"
  echo "=== [$(date '+%F %T')] START eval ${NAME}_final ==="
  python main.py manager.task_names=[ALL] trainer.train_bool=false trainer.test_bool=true \
    trainer.eval_n_times=200 manager.times_repeat=1 trainer.eval_checkpoint=_final \
    manager.parallel_eval=true manager.num_envs=10 \
    trainer.model_path="$ROOT/$NAME" wandb.run_name="${NAME}_final" \
    hydra.run.dir="$ROOT/eval/${NAME}_final"
  rc=$?
  echo "=== [$(date '+%F %T')] END eval ${NAME}_final (exit $rc) ==="
fi
echo "=== [$(date '+%F %T')] SCENE_KMEANS PIPELINE DONE ==="
