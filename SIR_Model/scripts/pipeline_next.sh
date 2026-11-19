#!/bin/bash
# Waits for the idea-6 eval, reads its 24-task mean, and launches whichever follow-up the
# number implies. Runs unattended overnight, so the branch is decided here rather than by
# a human reading the table in the morning.
#
# What idea 6 measures
#
#   SIR's recipe, unchanged, supervised toward VGIB's DISCOVERED node list instead of the
#   hand-written RELEVENT_NODES. Everything else is held fixed, so the result separates two
#   diagnoses that every previous VGIB number confounds:
#
#     result ~ SIR (16.58)   the discovered list is as good as the human one, and the whole
#                            VGIB deficit is the TRAINING MECHANISM -- noise injection, the
#                            soft-train/hard-eval split, the KL's pressure shape.
#     result <  SIR          the LIST itself is worse, and the mechanism that produces it is
#                            what needs fixing.
#
#   Reference points, all 24 tasks, seed 42: SIR 16.58 (seeds 43/44: 16.42/16.79, so sd is
#   ~0.19), VGIB soft 15.10, stage 2 14.56, noise 13.67, hard 10.54.
#
# The 0.5pp band below is ~2.6 sd of the SIR seed spread -- wide enough that a tie is
# genuinely a tie, narrow enough that the 1.5pp partial signal would not land in it.
set -u
cd /home/mattes/repos/sir
ROOT=/home/mattes/repos/sir/logs/robocasa_vgib
EVAL="$ROOT/eval/SIRvgiblist_s42_bs16_final"
SIR_REF=16.58
BAND=0.5

count_results() { ls "$EVAL"/eval_per_scene_*.json 2>/dev/null | wc -l; }

# --- wait for the eval ----------------------------------------------------------------
# All 24 result files are written in one burst at the very end, so the count is 0 until the
# run is essentially finished -- it is a completion flag, not a progress bar. The session
# check is what keeps this from waiting on a corpse: pipeline_idea6.sh ends by exiting,
# which takes the session with it, so "session gone AND fewer than 24 files" is a crash.
echo "=== [$(date '+%F %T')] waiting for the idea-6 eval ==="
until [ "$(count_results)" -eq 24 ]; do
  if ! tmux has-session -t sir_idea6 2>/dev/null; then
    sleep 30                                   # let a final flush land before judging
    if [ "$(count_results)" -eq 24 ]; then break; fi
    echo "=== [$(date '+%F %T')] ABORT: sir_idea6 gone at $(count_results)/24 results ==="
    exit 1
  fi
  sleep 120
done
echo "=== [$(date '+%F %T')] idea-6 eval complete ==="

# --- read the number and pick the branch ----------------------------------------------
MEAN=$(python3 - "$EVAL" <<'PY'
import json, glob, os, sys
rates = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "eval_per_scene_*.json"))):
    ps = json.load(open(f))["per_scene"]
    s = sum(v["n_success"] for v in ps.values())
    n = sum(v["n_rounds"] for v in ps.values())
    rates.append(100.0 * s / n)
print(f"{sum(rates)/len(rates):.2f}")
PY
)
echo "=== [$(date '+%F %T')] idea 6 (SIRvgiblist_s42_bs16) 24-task mean: $MEAN  vs SIR $SIR_REF ==="

if awk "BEGIN{exit !($MEAN >= $SIR_REF - $BAND)}"; then
  BRANCH=mechanism
  NAME=VGIB_s42_bs16_stage2_lr5e-5
  echo "=== [$(date '+%F %T')] BRANCH: mechanism -- the discovered list ties SIR, so the"
  echo "===          deficit is how VGIB trains, not what it picks. Stage 2 again at half"
  echo "===          the learning rate: 1e-4 recovered 88% of the hard-drop penalty and"
  echo "===          overshot the noise ceiling, which is the signature of too large a step"
  echo "===          on a warm start, not of a bad target."
else
  BRANCH=selection
  NAME=VGIB_s42_bs16_udw1e-1
  echo "=== [$(date '+%F %T')] BRANCH: selection -- the discovered list is worse than the"
  echo "===          human one, so fix what gets selected. The measured failure is score"
  echo "===          COMPRESSION, not mis-ranking: 19 tasks drop a human-labelled task"
  echo "===          object, median miss 0.081, 11 of 19 by <0.10, on a non-robot spread of"
  echo "===          only 0.2-0.4. uniform_dist_weight is the soft histogram loss that"
  echo "===          pushes scores to the ends of [0,1] without naming which node goes"
  echo "===          where -- distribution shaping, so the run stays unsupervised. SIR runs"
  echo "===          it at 1e-1; our VGIB config had it at 0."
fi

# --- GPU handoff ----------------------------------------------------------------------
# Never match on process names here: the tmux SERVER's cmdline is permanently whatever
# command first started it, so a pgrep guard once matched its own host and idled the GPU
# for 4h54m. nvidia-smi cannot be fooled that way.
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  echo "=== [$(date '+%F %T')] waiting: GPU busy ==="
  sleep 60
done

# --- train ----------------------------------------------------------------------------
echo "=== [$(date '+%F %T')] START train $NAME ($BRANCH) ==="
if [ -f "$ROOT/$NAME/action_generator_final.pth" ]; then
  echo "=== [$(date '+%F %T')] SKIP train $NAME (exists) ==="
elif [ "$BRANCH" = mechanism ]; then
  # Warm start from the same stage-1 checkpoint and the same k-means whitelist as the 1e-4
  # run, so lr is the single variable. pos_weight 3.69 is re-derived from THIS whitelist
  # (21.3% positives); the config's 4.3 assumed a smaller subgraph, and too high a value
  # flattens the scorer into an input-independent constant.
  python main.py manager.task_names=[ALL] manager.batch_size=16 \
    trainer.train_bool=true trainer.test_bool=false trainer.seed=42 \
    method/graph_encoder=xai_gnn_vgib_stage2 \
    method.graph_encoder.sparsification_layer.node_whitelist="$ROOT/VGIB_s42_bs16/subgraph_kmeans.yaml" \
    method.graph_encoder.sparsification_layer.whitelist_pos_weight=3.69 \
    method.optimization.lr=5e-5 \
    trainer.init_from="$ROOT/VGIB_s42_bs16" \
    wandb.run_name="$NAME" \
    hydra.run.dir="$ROOT/$NAME"
  rc=$?
  echo "=== [$(date '+%F %T')] END train $NAME (exit $rc) ==="
else
  # Stage 1 from scratch, seed 42, identical to VGIB_s42_bs16 except for the histogram
  # loss -- so it is directly comparable to that run's 15.10.
  python main.py manager.task_names=[ALL] manager.batch_size=16 \
    trainer.train_bool=true trainer.test_bool=false trainer.seed=42 \
    method/graph_encoder=xai_gnn_vgib \
    method.graph_encoder.sparsification_layer.uniform_dist_weight=1e-1 \
    wandb.run_name="$NAME" \
    hydra.run.dir="$ROOT/$NAME"
  rc=$?
  echo "=== [$(date '+%F %T')] END train $NAME (exit $rc) ==="
fi

# --- eval -----------------------------------------------------------------------------
# soft for the stage-1 arm: it is the only mode whose number reflects the learned scores
# alone rather than the scores plus a threshold chosen afterwards, which is exactly the
# thing under test. Stage 2 selects with its own probs > 0.5 and needs no threshold either.
if [ -n "$(ls "$ROOT/eval/${NAME}_final"/eval_per_scene_*.json 2>/dev/null)" ]; then
  echo "=== [$(date '+%F %T')] SKIP eval ${NAME}_final (results present) ==="
elif [ ! -f "$ROOT/$NAME/action_generator_final.pth" ]; then
  echo "=== [$(date '+%F %T')] ABORT eval: $NAME has no _final checkpoint (train failed) ==="
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
echo "=== [$(date '+%F %T')] PIPELINE NEXT DONE ($BRANCH / $NAME) ==="
