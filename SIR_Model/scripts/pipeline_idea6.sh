#!/bin/bash
# Idea 6: run the SIR recipe unchanged, but supervised toward VGIB's DISCOVERED node
# list instead of the hand-written RELEVENT_NODES.
#
# Why this is the experiment worth running next
#
#   Every VGIB number we have comes from a model VGIB also TRAINED -- noise injection, a
#   compression term that charges each node in proportion to how far it deviates from its
#   graph's mean, and a soft-train/hard-eval split. So "VGIB is 1.95pp behind SIR" cannot
#   distinguish two very different diagnoses:
#
#       (a) the discovered LIST is worse than the human one, or
#       (b) the list is fine and VGIB's TRAINING MECHANISM is what costs the performance.
#
#   Holding the recipe fixed and moving only its target separates them:
#       result ~ SIR   -> the list is fine; the mechanism is the problem (fix the loss)
#       result ~ VGIB  -> the list is the problem (fix what gets selected)
#   Either way half the search space dies, which no other queued run achieves.
#
# Handicap to keep in mind when reading it: RELEVENT_NODES is consulted as
# is_node_relevant_for_task(name, task, lang_goal), which on the four door tasks resolves
# the door named in THIS episode's instruction and takes k+1. A per-task list cannot
# express that, so those four tasks are penalised by the loss of language grounding
# independently of list quality. Read the door tasks separately from the rest.
set -u
cd /home/mattes/repos/sir
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib
WL="$ROOT/VGIB_s42_bs16/subgraph_kmeans.yaml"
NAME=SIRvgiblist_s42_bs16

# --- chain safely behind the running queue -------------------------------------------
# `sir_hard` is mid-flight and editing a live bash script corrupts it (bash reads by byte
# offset), so instead wait for the two seed-42 decomposition evals we still want, then
# retire that session before it starts seed 43.
echo "=== [$(date '+%F %T')] waiting for VGIB_s42_bs16_{hard,noise} to finish ==="
until [ -n "$(ls "$ROOT/eval/VGIB_s42_bs16_hard"/eval_per_scene_*.json 2>/dev/null)" ] \
   && [ -n "$(ls "$ROOT/eval/VGIB_s42_bs16_noise"/eval_per_scene_*.json 2>/dev/null)" ]; do
  sleep 300
done
echo "=== [$(date '+%F %T')] seed-42 decomposition complete; retiring sir_hard ==="
tmux kill-session -t sir_hard 2>/dev/null
sleep 10

while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  echo "=== [$(date '+%F %T')] waiting: GPU busy ==="
  sleep 60
done

# --- train ----------------------------------------------------------------------------
echo "=== [$(date '+%F %T')] START train $NAME ==="
if [ ! -f "$ROOT/$NAME/action_generator_final.pth" ]; then
  python main.py manager.task_names=[ALL] manager.batch_size=16 \
    trainer.train_bool=true trainer.test_bool=false trainer.seed=42 \
    method/graph_encoder=xai_gnn \
    method.graph_encoder.sparsification_layer.relevance_whitelist="$WL" \
    wandb.run_name="$NAME" \
    hydra.run.dir="$ROOT/$NAME"
  rc=$?
  echo "=== [$(date '+%F %T')] END train $NAME (exit $rc) ==="
else
  echo "=== [$(date '+%F %T')] SKIP train $NAME (exists) ==="
fi

# --- eval -----------------------------------------------------------------------------
# Evaluated exactly like the SIR baseline it is being compared against: same protocol,
# same checkpoint suffix, no per-task thresholds (top-k needs none).
if [ -n "$(ls "$ROOT/eval/${NAME}_final"/eval_per_scene_*.json 2>/dev/null)" ]; then
  echo "=== [$(date '+%F %T')] SKIP eval ${NAME}_final (results present) ==="
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
echo "=== [$(date '+%F %T')] IDEA 6 DONE ==="
