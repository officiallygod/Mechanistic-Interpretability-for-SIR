#!/bin/bash
# Two stage-2 arms on the udw stage-1 checkpoint, run back to back.
#
# Why the udw checkpoint and not VGIB_s42_bs16
#
#   `uniform_dist_weight=1e-1` fixed the compression that made the cut arbitrary. On the
#   val snapshot, seed 42, 20 non-door tasks, under the SHIPPING rule:
#
#                    recall   blind tasks   non-robot band   subgraph
#       VGIB_s42      25.0%      12 / 20        0.371          1.4
#       udw1e-1       79.5%       1 / 20        0.545          5.2
#
#   and it survives the size control -- at matched k the udw model wins at every k from 2
#   to 10 (peak +15.9pp at k=4), with the median rank of a human target inside the scene
#   ordering moving 4.0 -> 3.0. So the loss reordered the ranking, it did not merely
#   stretch it. Any stage-2 arm cut from the OLD checkpoint now tests a superseded
#   selection.
#
# The two arms
#
#   A  kmeans        the shipping cut, unchanged, on the repaired scores.  recall 79.5%
#   B  scene_kmeans  the cut fitted to the scene alone.                    recall 90.9%
#
#   A first, deliberately: if it lands near SIR then the whole fix was the training loss
#   and the cut rule never needed changing, which is the more parsimonious claim. B then
#   says what the better cut is worth on top of that. Running A alone would leave the
#   combination unmeasured; running B alone would confound the two fixes.
set -u
cd /home/mattes/repos/sir
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib
UDW="$ROOT/VGIB_s42_bs16_udw1e-1"

# name                       whitelist                        pos_weight (neg/pos, this list)
ARMS=(
  "VGIB_s42_udw_s2_kmeans    $UDW/subgraph_kmeans.yaml        1.40"
  "VGIB_s42_udw_s2_scenekm   $UDW/subgraph_scene_kmeans.yaml  1.12"
)

for spec in "${ARMS[@]}"; do
  set -- $spec
  [ -f "$2" ] || { echo "=== ABORT: missing whitelist $2 ==="; exit 1; }
done

# --- wait for the udw eval to release the GPU ------------------------------------------
echo "=== [$(date '+%F %T')] waiting for the udw eval ==="
until [ "$(ls "$ROOT/eval/VGIB_s42_bs16_udw1e-1_final"/eval_per_scene_*.json 2>/dev/null | wc -l)" -eq 24 ]; do
  if ! tmux has-session -t sir_next 2>/dev/null; then
    sleep 30
    n=$(ls "$ROOT/eval/VGIB_s42_bs16_udw1e-1_final"/eval_per_scene_*.json 2>/dev/null | wc -l)
    echo "=== [$(date '+%F %T')] sir_next gone at $n/24 udw eval results; continuing to stage 2 anyway ==="
    break
  fi
  sleep 120
done

run_arm() {
  NAME=$1; WL=$2; PW=$3
  # Never match on process names when waiting for the GPU: the tmux SERVER's cmdline is
  # permanently the command that first started it, so a name-based guard once matched its
  # own host and idled the GPU for 4h54m. nvidia-smi cannot be fooled that way.
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    echo "=== [$(date '+%F %T')] waiting: GPU busy (before $NAME) ==="
    sleep 60
  done

  echo "=== [$(date '+%F %T')] START train $NAME  (whitelist $(basename "$WL"), pos_weight $PW) ==="
  if [ -f "$ROOT/$NAME/action_generator_final.pth" ]; then
    echo "=== [$(date '+%F %T')] SKIP train $NAME (exists) ==="
  else
    # pos_weight is re-derived from THIS whitelist, not carried over: too high and the BCE
    # flattens the scorer into an input-independent constant, and
    # sparsification/{train,val}_score_std_* is the canary -- it collapses toward 0 an epoch
    # or two before the means visibly move.
    python main.py manager.task_names=[ALL] manager.batch_size=16 \
      trainer.train_bool=true trainer.test_bool=false trainer.seed=42 \
      method/graph_encoder=xai_gnn_vgib_stage2 \
      method.graph_encoder.sparsification_layer.node_whitelist="$WL" \
      method.graph_encoder.sparsification_layer.whitelist_pos_weight="$PW" \
      trainer.init_from="$UDW" \
      wandb.run_name="$NAME" \
      hydra.run.dir="$ROOT/$NAME"
    rc=$?
    echo "=== [$(date '+%F %T')] END train $NAME (exit $rc) ==="
  fi

  # Stage 2 selects with its own scalar probs > 0.5 -- the BCE drives scores to the
  # extremes, so no per-task threshold is passed. If stage 2 still needs one, it did not do
  # its job.
  if [ -n "$(ls "$ROOT/eval/${NAME}_final"/eval_per_scene_*.json 2>/dev/null)" ]; then
    echo "=== [$(date '+%F %T')] SKIP eval ${NAME}_final (results present) ==="
  elif [ ! -f "$ROOT/$NAME/action_generator_final.pth" ]; then
    echo "=== [$(date '+%F %T')] ABORT eval: $NAME has no _final checkpoint ==="
    return 1
  else
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
      echo "=== [$(date '+%F %T')] waiting: GPU busy (before eval $NAME) ==="
      sleep 60
    done
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
}

for spec in "${ARMS[@]}"; do
  set -- $spec
  run_arm "$1" "$2" "$3" || echo "=== [$(date '+%F %T')] arm $1 failed; continuing to the next ==="
done
echo "=== [$(date '+%F %T')] STAGE2 ARMS DONE ==="
