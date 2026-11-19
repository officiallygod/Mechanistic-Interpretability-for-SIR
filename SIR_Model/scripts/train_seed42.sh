#!/bin/bash
# Seed-42 arm of the VGIB-vs-SIR comparison: three ALL-task trainings, strictly one at a
# time (each needs ~8 GB VRAM and ~36 GB RAM; running two would contend for both).
#
# Training only -- trainer.test_bool=false. Evaluation is a separate eval-only run per
# checkpoint, which is what lets one VGIB checkpoint be scored under several
# eval_prune_modes without retraining, and gives every eval its own wandb run.
#
# Output dirs are pinned with hydra.run.dir instead of the default timestamp, so the
# eval step can name a checkpoint rather than hunt for it.
set -u
cd /home/mattes/repos/sir

SEED=42
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib
COMMON=(manager.task_names=[ALL] trainer.seed=$SEED
        trainer.train_bool=true trainer.test_bool=false)

run () {
  local name=$1; shift
  local dir="$ROOT/$name"
  if [ -f "$dir/action_generator_final.pth" ]; then
    echo "=== [$(date +%H:%M:%S)] SKIP $name (already trained) ==="; return 0
  fi
  echo "=== [$(date +%H:%M:%S)] START $name ==="
  # No `set -e`: one failing arm must not cost the other two a night of GPU time.
  python main.py "${COMMON[@]}" "$@" hydra.run.dir="$dir"
  echo "=== [$(date +%H:%M:%S)] END $name (exit $?) ==="
}

run "SIR_s$SEED"   method/graph_encoder=xai_gnn
run "VGIB_s$SEED"  method/graph_encoder=xai_gnn_vgib
run "Image_s$SEED" 'manager.graph_modalities=[]' 'manager.image_modalities=[left,right]'

echo "=== [$(date +%H:%M:%S)] ALL SEED-$SEED TRAININGS DONE ==="
for n in "SIR_s$SEED" "VGIB_s$SEED" "Image_s$SEED"; do
  printf "  %-12s %s\n" "$n" \
    "$([ -f "$ROOT/$n/action_generator_final.pth" ] && echo OK || echo MISSING)"
done
