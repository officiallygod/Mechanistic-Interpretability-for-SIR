#!/bin/bash
# Evaluation matrix for one seed. Run as `scripts/eval_seed.sh 42`, after
# scripts/train_seed.sh has produced the checkpoints.
#
# Every row is an eval-only run: trainer.train_bool=false makes Trainer reload the
# checkpoint's own manager/method config, so the architecture comes from the checkpoint
# and only the eval-time knobs come from this command line. That is what lets one VGIB
# checkpoint be scored under several eval_prune_modes without retraining.
#
# VGIB's `threshold` is a pure eval-time knob (threshold_sum_weight is 0.0, so it never
# enters a gradient), so it is chosen after training from the model's own val scores:
#   python -m utils.suggest_threshold logs/robocasa_vgib/VGIB_s42/node_scores/val_epoch049.json
#
# 0.70 is that choice for seed 42, and it is NOT the number that tool suggests. Its
# median-of-per-task-gaps was 0.847, which leaves OpenDrawer with an EMPTY subgraph --
# the pooled graph embedding becomes zero, the policy goes scene-blind, and the task
# scores 0 for a reason that has nothing to do with VGIB. Six more tasks drop to a single
# node there. Scanning global cutoffs instead: 0.70 keeps 34% of nodes, which is exactly
# the 0.34 SIR's own top-k logged, with a floor of 3 nodes on every task and no empties.
# So it is sparsity-matched to the baseline while reading nothing from RELEVENT_NODES.
#
# Re-derive it per seed. A cut read off seed 42's scores is not valid for seed 43.
set -u
SEED=${1:?usage: eval_seed.sh <seed>}
VGIB_THRESHOLD=${VGIB_THRESHOLD:-0.70}

cd /home/mattes/repos/sir
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib
EVAL=(manager.task_names=[ALL] trainer.train_bool=false trainer.test_bool=true
      trainer.eval_n_times=200 manager.times_repeat=1
      manager.parallel_eval=true manager.num_envs=10)

run () {
  local ckpt=$1 name=$2; shift 2
  if [ ! -f "$ROOT/$ckpt/action_generator_final.pth" ]; then
    echo "=== [$(date +%H:%M:%S)] SKIP $name (no checkpoint at $ROOT/$ckpt) ==="; return 0
  fi
  echo "=== [$(date +%H:%M:%S)] START $name ==="
  python main.py "${EVAL[@]}" trainer.model_path="$ROOT/$ckpt" \
    wandb.run_name="$name" hydra.run.dir="$ROOT/eval/$name" "$@"
  echo "=== [$(date +%H:%M:%S)] END $name (exit $?) ==="
  # A CUDA OOM in the policy server is answered with zero actions rather than a crash,
  # so the run completes and just reads low. Surface it here too.
  grep -q "DEGRADED EVALUATION" "$ROOT/eval/$name/main.log" 2>/dev/null \
    && echo "!!! $name: DEGRADED EVALUATION -- number is a lower bound, drop num_envs !!!"
}

THR=()
[ -n "$VGIB_THRESHOLD" ] && THR=(method.graph_encoder.sparsification_layer.threshold=$VGIB_THRESHOLD)

run "SIR_s$SEED"   "SIR_s$SEED"
run "Image_s$SEED" "Image_s$SEED"
# soft = what the learned scores alone are worth (no threshold at all).
run "VGIB_s$SEED"  "VGIB_s${SEED}_soft" \
    method.graph_encoder.sparsification_layer.eval_prune_mode=soft "${THR[@]}"
# hard = the shippable pruned subgraph, and the like-for-like comparison against SIR,
# which also prunes hard.
run "VGIB_s$SEED"  "VGIB_s${SEED}_hard" \
    method.graph_encoder.sparsification_layer.eval_prune_mode=hard "${THR[@]}"

echo "=== [$(date +%H:%M:%S)] ALL SEED-$SEED EVALS DONE ==="
