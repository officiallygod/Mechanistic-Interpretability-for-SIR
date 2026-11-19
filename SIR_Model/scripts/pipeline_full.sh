#!/bin/bash
# Everything remaining, in one detached chain. Strictly one job at a time: training needs
# ~8 GB VRAM, evaluation ~27 GB, so overlapping them OOMs the policy server -- which does
# not crash, it answers with zero actions and quietly understates the run.
#
# Phase 1  VGIB seed 42 at batch 16, soft eval only. VGIB was the one arm still trained at
#          batch 128, so until this exists the VGIB-vs-SIR comparison rests on an
#          inference (batch size did not move SIR: +0.23pp, 95% CI [-0.75,+1.25]) rather
#          than on a measurement of VGIB itself.
# Phase 2  Seeds 43 and 44 for all three arms at batch 16, _final checkpoints, VGIB soft.
#          Ordered seed-by-seed so an interruption leaves a COMPLETE extra seed rather
#          than three-quarters of two.
#
# Only `_final` is evaluated: best-val selection was measured and does not help
# (SIR -0.31pp, Image -0.62pp, both n.s.) -- val MSE is near-decoupled from rollout
# success here. `_best` is still written, it is just not scored.
set -u
cd /home/mattes/repos/sir
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib

train () {                       # train <name> <hydra overrides...>
  local name=$1; shift
  local dir="$ROOT/$name"
  if [ -f "$dir/action_generator_final.pth" ]; then
    echo "=== [$(date '+%F %T')] SKIP train $name (exists) ==="; return 0
  fi
  echo "=== [$(date '+%F %T')] START train $name ==="
  python main.py manager.task_names=[ALL] manager.batch_size=16 \
    trainer.train_bool=true trainer.test_bool=false hydra.run.dir="$dir" "$@"
  rc=$?
  echo "=== [$(date '+%F %T')] END train $name (exit $rc) ==="
}

evaluate () {                    # evaluate <ckpt dir> <run name> <extra overrides...>
  local ckpt=$1 name=$2; shift 2
  if [ ! -f "$ROOT/$ckpt/action_generator_final.pth" ]; then
    echo "=== [$(date '+%F %T')] SKIP eval $name (no checkpoint) ==="; return 0
  fi
  # Test for RESULTS, not for the directory. A crashed eval still leaves its hydra run
  # dir behind -- the 2026-09-06 wandb outage left two of them empty -- and a
  # directory-existence guard would then skip the retry and silently keep the hole.
  if [ -n "$(ls "$ROOT/eval/$name"/eval_per_scene_*.json 2>/dev/null)" ]; then
    echo "=== [$(date '+%F %T')] SKIP eval $name (results present) ==="; return 0
  fi
  rm -rf "$ROOT/eval/$name"
  echo "=== [$(date '+%F %T')] START eval $name ==="
  python main.py manager.task_names=[ALL] trainer.train_bool=false trainer.test_bool=true \
    trainer.eval_n_times=200 manager.times_repeat=1 trainer.eval_checkpoint=_final \
    manager.parallel_eval=true manager.num_envs=10 \
    trainer.model_path="$ROOT/$ckpt" wandb.run_name="$name" \
    hydra.run.dir="$ROOT/eval/$name" "$@"
  rc=$?
  echo "=== [$(date '+%F %T')] END eval $name (exit $rc) ==="
  grep -q "DEGRADED EVALUATION" "$ROOT/eval/$name/main.log" 2>/dev/null \
    && echo "!!! $name: DEGRADED EVALUATION -- lower bound, drop num_envs !!!"
}

eval_vgib () {                   # soft eval; threshold only sets the REPORTED sparsity
  local ckpt=$1 name=$2
  local snap="$ROOT/$ckpt/node_scores/val_epoch049.json"
  local thr=0.70
  if [ -f "$snap" ]; then
    # Re-derived per seed on purpose. Under `soft` nothing is pruned, so the threshold
    # does not touch the policy -- but it does decide which nodes are REPORTED as
    # selected, and therefore the faithfulness numbers. The score distribution shifts
    # between seeds, so a fixed cutoff would keep 90% of nodes on one seed and 5% on
    # another and make those numbers incomparable. Matching kept-fraction to SIR's 0.34
    # keeps them on the same footing.
    thr=$(python -m utils.match_sparsity_threshold "$snap" --target 0.34 --verbose 2>&1 | tail -1)
    echo "--- $name: sparsity-matched threshold $thr (from its own val scores) ---"
  else
    echo "--- $name: no snapshot at $snap, falling back to $thr ---"
  fi
  evaluate "$ckpt" "$name" \
    method.graph_encoder.sparsification_layer.eval_prune_mode=soft \
    method.graph_encoder.sparsification_layer.threshold="$thr"
}

# ---------------- Phase 1: VGIB seed 42 at batch 16 ----------------
echo "=== [$(date '+%F %T')] PHASE 1: VGIB_s42_bs16 ==="
train    "VGIB_s42_bs16" trainer.seed=42 method/graph_encoder=xai_gnn_vgib
eval_vgib "VGIB_s42_bs16" "VGIB_s42_bs16_soft"

# ---------------- Phase 2: seeds 43 and 44, all three arms ----------------
for SEED in 43 44; do
  echo "=== [$(date '+%F %T')] PHASE 2: seed $SEED ==="
  train "SIR_s${SEED}_bs16"   trainer.seed=$SEED method/graph_encoder=xai_gnn
  train "VGIB_s${SEED}_bs16"  trainer.seed=$SEED method/graph_encoder=xai_gnn_vgib
  train "Image_s${SEED}_bs16" trainer.seed=$SEED \
        'manager.graph_modalities=[]' 'manager.image_modalities=[left,right]'

  evaluate  "SIR_s${SEED}_bs16"   "SIR_s${SEED}_bs16_final"
  eval_vgib "VGIB_s${SEED}_bs16"  "VGIB_s${SEED}_bs16_soft"
  evaluate  "Image_s${SEED}_bs16" "Image_s${SEED}_bs16_final"
done

echo "=== [$(date '+%F %T')] FULL PIPELINE DONE ==="
for n in VGIB_s42_bs16_soft \
         SIR_s43_bs16_final VGIB_s43_bs16_soft Image_s43_bs16_final \
         SIR_s44_bs16_final VGIB_s44_bs16_soft Image_s44_bs16_final; do
  printf "  %-26s %s\n" "$n" "$([ -d "$ROOT/eval/$n" ] && echo done || echo MISSING)"
done
