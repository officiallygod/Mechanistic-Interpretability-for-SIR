#!/bin/bash
# Phase 1  Stage-2 training on seed 42 (lr 1e-4), then its evaluation.
# Phase 2  Prune-mode decomposition of the three bs16 VGIB seeds.
#
# Stage 2 runs FIRST on purpose: it is the experiment that decides whether this
# direction works, while phase 2 is read-only diagnostics on checkpoints that are not
# going anywhere. Ordered the other way, the decisive number would be two days out.
#
# ---------------------------------------------------------------------------------
# Phase 1 -- stage 2
#
# The whitelist is stage 1's OWN unsupervised selection (1D 2-means per task on its own
# node scores), never RELEVENT_NODES. The BCE target is 0 for exactly the dropped nodes
# and 1 for exactly the kept ones -- a SELECTION-ALIGNED counterforce, which is what the
# compression KL is not. That is why a plain soft->hard anneal collapses to keep-all here
# (Heilinger thesis Sec. 4.2.3: "the scorer received a strong signal to push the node
# scores of the affected nodes up") while this does not: under a hard drop the STE tells
# every excluded node "you would have helped", and only a per-node target aimed at the
# same cut can hold against it.
#
# ---------------------------------------------------------------------------------
# Phase 2 -- why three prune modes and not just `hard`
#
#   soft   (already done)  nothing is pruned          -> the no-pruning ceiling
#   noise                  the SAME nodes are cut, but replaced by a draw from the
#                          graph's own Gaussian; topology, edges and pooling weights
#                          are exactly what training used  -> isolates INFORMATION LOSS
#   hard                   nodes zeroed and edges cut -> adds DISTRIBUTION SHIFT
#
# soft-noise is what the subgraph was carrying; noise-hard is what the policy loses
# purely from never having seen a pruned graph. Only the second is what stage 2 repairs,
# so this bounds what stage 2 could have bought. (Thesis Sec. 5.3.3 measures 19.7 -> 26.2
# on 4 tasks; this is that experiment on 24.)
#
# Thresholds are PER TASK, from utils.select_subgraph --kmeans on each seed's own stage-1
# scores. A single global bar cannot work: the fitted per-task cutoffs span 0.64-0.87, so
# one bar necessarily over-prunes one task and under-prunes another. The same file also
# carries the node lists, so eval and stage 2 are driven by one artifact and cannot
# silently disagree.
set -u
cd /home/mattes/repos/sir
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib

evaluate () {                    # evaluate <ckpt dir> <run name> <extra overrides...>
  local ckpt=$1 name=$2; shift 2
  if [ ! -f "$ROOT/$ckpt/action_generator_final.pth" ]; then
    echo "=== [$(date '+%F %T')] SKIP eval $name (no checkpoint) ==="; return 0
  fi
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

eval_pruned () {                 # eval_pruned <seed> <hard|noise>
  local seed=$1 mode=$2
  local ckpt="VGIB_s${seed}_bs16"
  local wl="$ROOT/$ckpt/subgraph_kmeans.yaml"
  if [ ! -f "$wl" ]; then
    echo "!!! no $wl -- run: python -m utils.select_subgraph $ROOT/$ckpt/node_scores/val_epoch049.json --kmeans -o $wl"
    return 1
  fi
  evaluate "$ckpt" "${ckpt}_${mode}" \
    method.graph_encoder.sparsification_layer.eval_prune_mode="$mode" \
    trainer.task_thresholds="$wl"
}

# Evaluation needs ~27 GB; two concurrent jobs do not crash, they silently answer with
# zero actions and understate the run. Wait any other run out.
#
# Wait on the actual GPU consumer -- a `python main.py` -- and NOT on the driving script
# name. The tmux SERVER's cmdline is permanently the command that first started it
# ("tmux new-session -d -s sir_full bash scripts/pipeline_full.sh"), and it outlives that
# session for as long as any other session exists. A `pgrep -f pipeline_full.sh` guard
# therefore matches the very server hosting this script and waits forever: on 2026-09-08
# it held the GPU idle for 4h54m after pipeline_full had already finished.
# Ask the GPU who is on it rather than matching process names at all -- no string can
# self-match, and it is the resource we actually contend for.
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  echo "=== [$(date '+%F %T')] waiting: GPU busy (pids $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')) ==="
  sleep 300
done
echo "=== [$(date '+%F %T')] GPU free, starting ==="

# ---------------- Phase 1: stage 2, seed 42 ----------------
echo "=== [$(date '+%F %T')] PHASE 1: stage 2, seed 42, lr 1e-4 ==="
# pos_weight is re-derived from THIS whitelist (21.3% positives -> neg/pos 3.69), not the
# config's 4.3, which assumed a smaller subgraph. Too high and the BCE flattens the
# scorer into an input-independent constant; sparsification/{train,val}_score_std_* is
# the canary and collapses toward 0 an epoch or two before the means visibly move.
S2=VGIB_s42_bs16_stage2_lr1e-4
if [ ! -f "$ROOT/$S2/action_generator_final.pth" ]; then
  echo "=== [$(date '+%F %T')] START train $S2 ==="
  python main.py manager.task_names=[ALL] manager.batch_size=16 \
    trainer.train_bool=true trainer.test_bool=false trainer.seed=42 \
    method/graph_encoder=xai_gnn_vgib_stage2 \
    method.graph_encoder.sparsification_layer.node_whitelist="$ROOT/VGIB_s42_bs16/subgraph_kmeans.yaml" \
    method.graph_encoder.sparsification_layer.whitelist_pos_weight=3.69 \
    trainer.init_from="$ROOT/VGIB_s42_bs16" \
    wandb.run_name="$S2" \
    hydra.run.dir="$ROOT/$S2"
  rc=$?
  echo "=== [$(date '+%F %T')] END train $S2 (exit $rc) ==="
else
  echo "=== [$(date '+%F %T')] SKIP train $S2 (exists) ==="
fi

# Stage 2 selects with its own scalar `probs > 0.5`: the BCE drives the scores to the
# extremes, so the bar stops being a tuned quantity. No per-task thresholds here on
# purpose -- if stage 2 still needs them, it did not do its job.
evaluate "$S2" "${S2}_final"

# ---------------- Phase 2: prune-mode decomposition ----------------
echo "=== [$(date '+%F %T')] PHASE 2: prune-mode decomposition ==="
for SEED in 42 43 44; do          # seed-major: an interruption leaves whole seeds
  eval_pruned $SEED hard
  eval_pruned $SEED noise
done

echo "=== [$(date '+%F %T')] ALL DONE ==="
