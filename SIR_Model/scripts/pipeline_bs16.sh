#!/bin/bash
# Batch-size-16 reconciliation arm, queued behind the running seed-42 eval chain.
#
# Why: every archived run in this repo used batch_size 16; the 24-task runs used 128,
# which is the one training variable that changed. The LR scheduler steps per EPOCH, so
# at 50 epochs the LR curve is identical but the number of optimiser updates drops ~8x
# (~90k vs ~725k) with lr still at the 1e-4 tuned for batch 16. This arm isolates that.
#
# Both checkpoints are now written by every training run and both get evaluated:
#   _final — last epoch
#   _best  — lowest val_total_loss (the 24-task image baseline peaked at epoch 9 and
#            then overfitted for 40 more epochs, so these can differ a lot)
#
# Runs strictly one job at a time: training needs ~8 GB VRAM, evaluation ~27 GB.
set -u
cd /home/mattes/repos/sir
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib
SEED=42

# ---- wait for the seed-42 eval chain (VGIB soft, then VGIB hard) to finish ----
echo "=== [$(date '+%F %T')] waiting for the running eval chain ==="
while pgrep -f "scripts/eval_seed.sh $SEED" > /dev/null; do sleep 60; done
echo "=== [$(date '+%F %T')] eval chain done; starting batch-16 retraining ==="

train () {
  local name=$1; shift
  local dir="$ROOT/$name"
  if [ -f "$dir/action_generator_final.pth" ]; then
    echo "=== [$(date '+%F %T')] SKIP train $name (exists) ==="; return 0
  fi
  echo "=== [$(date '+%F %T')] START train $name ==="
  python main.py manager.task_names=[ALL] trainer.seed=$SEED manager.batch_size=16 \
    trainer.train_bool=true trainer.test_bool=false hydra.run.dir="$dir" "$@"
  echo "=== [$(date '+%F %T')] END train $name (exit $?) ==="
}

evaluate () {
  local ckpt_dir=$1 suffix=$2 name=$3; shift 3
  if [ ! -f "$ROOT/$ckpt_dir/action_generator${suffix}.pth" ]; then
    echo "=== [$(date '+%F %T')] SKIP eval $name (no ${suffix} checkpoint) ==="; return 0
  fi
  echo "=== [$(date '+%F %T')] START eval $name ==="
  python main.py manager.task_names=[ALL] trainer.train_bool=false trainer.test_bool=true \
    trainer.eval_n_times=200 manager.times_repeat=1 \
    manager.parallel_eval=true manager.num_envs=10 \
    trainer.eval_checkpoint="$suffix" trainer.model_path="$ROOT/$ckpt_dir" \
    wandb.run_name="$name" hydra.run.dir="$ROOT/eval/$name" "$@"
  echo "=== [$(date '+%F %T')] END eval $name (exit $?) ==="
  grep -q "DEGRADED EVALUATION" "$ROOT/eval/$name/main.log" 2>/dev/null \
    && echo "!!! $name: DEGRADED EVALUATION -- lower bound, drop num_envs !!!"
}

train "SIR_s${SEED}_bs16"   method/graph_encoder=xai_gnn
train "Image_s${SEED}_bs16" 'manager.graph_modalities=[]' 'manager.image_modalities=[left,right]'

for m in SIR Image; do
  for ck in _final _best; do
    evaluate "${m}_s${SEED}_bs16" "$ck" "${m}_s${SEED}_bs16${ck}"
  done
done

echo "=== [$(date '+%F %T')] BATCH-16 PIPELINE DONE ==="
for m in SIR Image; do for ck in _final _best; do
  n="${m}_s${SEED}_bs16${ck}"
  printf "  %-26s %s\n" "$n" "$([ -d "$ROOT/eval/$n" ] && echo done || echo MISSING)"
done; done
