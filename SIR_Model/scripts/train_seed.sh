#!/bin/bash
# Generalised form of scripts/train_seed42.sh -- run as `scripts/train_seed.sh 43`.
# Kept as a separate file because bash reads a script lazily: editing train_seed42.sh
# while it was running would have corrupted the in-flight run.
#
# One training at a time (~8 GB VRAM, ~36 GB RAM each). Training only; evaluation is a
# separate eval-only run per checkpoint.
set -u
SEED=${1:?usage: train_seed.sh <seed>}
cd /home/mattes/repos/sir

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
