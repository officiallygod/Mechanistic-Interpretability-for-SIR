# RoboCasa — VGIB / SIR results

Machine-readable record for paper drafting. Every number below is measured in this repo
unless marked otherwise. Last updated 2026-09-11.

---

## 1. Setup

| | |
|---|---|
| Benchmark | RoboCasa, **24 kitchen tasks**, single-arm PandaMobile |
| Policy | MDT diffusion action head, `act_window=10` |
| Graph encoder | 2-layer `WeightedGATv2`, hidden 512, 4 heads, mean pooling |
| Graph modalities | `cropped_image_feature`, `bb_coordinates` |
| Training | 50 epochs, AdamW lr `1e-4`, wd `1e-3`, batch 16, EMA on (`use_ema`, `init_from_ema`) |
| Checkpoint evaluated | `_final` (not `_best`) |
| **Rollouts per task** | **200 unique scenes × 1 repeat = 200 rollouts** (`eval_n_times=200`, `times_repeat=1`, `num_envs=10`) |
| **Rollouts per run** | **4800** (24 tasks × 200) |
| Seeds | 42 / 43 / 44 (`trainer.seed`, controls both model init and eval scene sampling) |

**Noise floor.** Per-task seed-to-seed spread at this protocol is ≤4.5pp (measured on
`baseline_ema_800`, the first config with no task above that), so a difference between two
runs on one task carries ~±6.4pp. The 24-task mean is far tighter — observed seed sd is
0.17–0.45pp. **Treat 24-task means as significant and single-task deltas below ~6pp as
noise.** More *unique* scenes is the only thing that shrinks this (1/√n); repeats do not.

---

## 2. Methods compared

| name | what it is | selection supervision |
|---|---|---|
| **SIR** | `topk_PerTask` + CBM loss (`use_cbm_loss=nodes_att`, `lcbm_weight=0.1`) + soft histogram loss (`uniform_dist_weight=0.1`) | **supervised** — `RELEVENT_NODES`, a hand-written per-task table; door tasks additionally resolve the door from each episode's language via `is_node_relevant_for_task` |
| **VGIB** | Variational Graph Information Bottleneck (Yu et al., CVPR 2022) as sparsifier. `vgib_noise_kl_mode=truncated`, `vgib_noise_kl_weight=1e-2`, `use_cbm_loss=null` | **unsupervised** — nothing on this path reads `RELEVENT_NODES` |
| **Image** | no graph; `image_modalities=[left,right]` CNN only | n/a |

VGIB training never prunes: node *v*'s features are blended into the graph's own Gaussian
`N(μ_G, σ_G²)` with weight `(1−p_v)`. Truncated KL has an interior optimum
`p* = 1/(1+D_i)` where `D_i` is *v*'s mean squared deviation from the graph mean.

**`eval_prune_mode`** (VGIB only) — decomposes the cost of pruning:
- `soft` — nothing pruned; attenuation toward the graph mean, as in training. Reflects the scores alone.
- `noise` — sub-threshold nodes replaced by graph-Gaussian draws; topology intact.
- `hard` — sub-threshold nodes zeroed **and edges cut**. The subgraph you would ship as the explanation.

---

## 3. Headline results (24-task mean success %, 4800 rollouts per cell)

| method | s42 | s43 | s44 | **mean ± sd** |
|---|---|---|---|---|
| **SIR** | 16.58 | 16.42 | 16.79 | **16.60 ± 0.19** |
| **VGIB (soft)** | 15.10 | 14.62 | 14.21 | **14.65 ± 0.45** |
| **Image only** | 12.83 | 13.12 | 12.83 | **12.93 ± 0.17** |

**SIR − VGIB = 1.95pp.** Both beat image-only.

Same three methods over the **16 non-PnP tasks** (the 8 PnP tasks sit at the 0% floor for
every method and drag all means toward zero — see §9.6):

| method | 16-task mean |
|---|---|
| SIR | **24.81** |
| VGIB (soft) | **21.90** |
| Image only | **19.08** |

### Single-seed arms (seed 42 only — no replicates)

| run | 24-task mean |
|---|---|
| VGIB soft | 15.10 |
| **VGIB + `uniform_dist_weight=1e-1`** (soft eval) | 15.06 |
| **VGIB stage 2** (BCE toward whitelist, lr 1e-4) | 14.56 |
| **VGIB stage 2 on the REPAIRED whitelist** (arm A, udw + kmeans) | 13.92 |
| **VGIB stage 2 on the BEST whitelist** (arm B, udw + scene_kmeans) | 14.31 |
| VGIB noise | 13.67 |
| VGIB hard | 10.54 |
| **SIR recipe trained on VGIB's discovered list** ("idea 6") | 14.77 |

---

## 4. Prune-mode decomposition (seed 42, 24 tasks)

```
soft  − noise  = 1.44pp   information the dropped nodes actually carried
noise − hard   = 3.12pp   pure distribution shift (policy never saw holes)
soft  − hard   = 4.56pp   total pruning penalty
stage2 − hard  = 4.02pp   repaired by stage 2 (88% of the penalty)
stage2 − noise = 0.90pp   stage 2 lands ABOVE the in-distribution ceiling
SIR − stage2   = 2.02pp   residual: selection quality
```

**Claim: two-thirds of the cost of hard pruning is distribution shift, not lost
information.** Training on pruned graphs (stage 2) recovers 88% of it and beats a policy
that never had to cope with holes.

*Caveat: single seed. The 3.12 / 4.02 terms are far above VGIB's seed sd (0.45); the
2.02pp residual is ~4.5× it — actionable, not publication-grade without seeds 43/44.*

---

## 5. Is the human node list optimal? (the "idea 6" experiment)

**Design.** SIR's recipe held completely fixed; only the CBM supervision target swapped
from `RELEVENT_NODES` to VGIB's discovered per-task list. Separates "the list is worse"
from "VGIB's training mechanism is worse".

**Result: 14.77 vs SIR 16.58 (−1.81pp), wins 5/24, loses 12/24.** The list is worse — but
not diffusely:

| | vgiblist | SIR | Δ |
|---|---|---|---|
| TurnOffSinkFaucet + TurnOnSinkFaucet | 31.50 | 44.75 | **−13.25** |
| other 22 tasks | 13.25 | 14.02 | **−0.77** |

On 22 of 24 tasks an **unsupervised** list matches the hand-written one to within 0.77pp.
It collapses only where it drops the manipulated object entirely — VGIB's list for
TurnOffSinkFaucet is `{PandaMobile, PandaGripper}`, with no `Sink`.

**Do not over-attribute to the sink tasks.** TurnSinkSpout is the built-in control: same
`Sink` removal, only −4.0pp, which is inside noise. The −14.5 / −12.0 cells clear ~2σ; the
"61% of the deficit" framing does not survive the control.

### Where VGIB *improves* on the human list

| task | Δ | VGIB added | VGIB dropped |
|---|---|---|---|
| CloseDrawer | +3.0 | `drawer_obj` | `Counter` |
| TurnOffStove | +2.5 | `cookware` | `Stove` |
| TurnOnMicrowave | +2.5 | — | `Microwave` |
| TurnOffMicrowave | +1.5 | — | `Microwave` |

Individually inside noise; the **pattern** is not. Unsupervised selection finds
manipulanda the human table omits (`drawer_obj`, `cookware`) and discards furniture it
includes. `Counter` is dropped in 12 tasks: Δ ≥ 0 in 8 of them (six of those are PnP tasks
at the 0% floor, so uninformative), −0.5 in two, and −4.0 / −4.5 on CoffeeServeMug and
OpenDrawer. Net evidence that `Counter` is droppable is weak, not clean.

### Objects that don't matter

Dropping the named object **helps** on TurnOnMicrowave (+2.5), TurnOffMicrowave (+1.5),
TurnOffStove (+2.5). Independent offline LOO (val loss, separate machine) agrees: objects
matter for **doors and PnP**, not the rest. *Caution: val loss in this repo is nearly blind
to sparsification — a documented case moved val MSE 0.058 → 0.078 while rollout success
moved 0.720 → 0.185. Prefer rollout evidence.*

### Door tasks are structurally handicapped for any per-task list

SIR's static entry for `OpenSingleDoor` is `{PandaMobile, PandaGripper}`; the door itself is
resolved **per episode from the instruction**. A static per-task list cannot express this.
**Exclude the 4 door tasks from list-quality comparisons**, or report them separately.

---

## 6. The selection rule discards a good ranking

**Read §6b before acting on this section.** Everything below is about agreement with the
human list, and §6b shows that fixing it moved success by −0.04pp. The analysis stands as a
description of the sparsifier; it is not evidence about task performance.

VGIB's *ranking* is good. Median rank of a human target inside VGIB's own ordering is **5**;
33/44 targets sit in the top 8 (`Drawer` #2 on CloseDrawer, `obj` #3–5 on every PnP,
`Sink` #4 on TurnOffSinkFaucet). The **cut** throws it away: `PandaMobile` scores ~0.99
while the whole scene occupies 0.45–0.76, so a 1D 2-means fitted to all nodes splits
robot-vs-scene and stops at 2–3 nodes.

- rank #1 is a robot node on **23/24** tasks; ranks #1 **and** #2 both robot on **15/24**
- robot share of all kept nodes: **53%** (45/85)
- **5/24** discovered subgraphs contain nothing but robot nodes

### Selection rules scored against `RELEVENT_NODES` (val snapshots, 3 seeds, 20 non-door tasks, robot excluded from every subgraph)

| rule | size | recall | prec | **tasks with 0 targets** |
|---|---|---|---|---|
| **all_kmeans (currently shipped)** | 1.6 | 27% | 36% | **11.0 / 20** |
| **scene_kmeans** (2-means fitted to the scene alone) | 4.0 | 56% | 31% | **4.0 / 20** |
| scene_maxgap | 4.0 | 46% | 25% | 6.7 / 20 |
| scene_relgap | 8.7 | 87% | 22% | 2.0 / 20 |
| scene_top3 | 3.0 | 46% | 34% | 4.3 / 20 |
| scene_top5 | 5.0 | 66% | 29% | 1.7 / 20 |
| scene_top6 | 6.0 | 73% | 27% | 1.0 / 20 |
| scene_top8 | 8.0 | 88% | 24% | 1.0 / 20 |

**The headline number is the last column: the rule we currently ship produces a subgraph
containing no task-relevant object at all on 11 of 20 tasks.**

Two methodological warnings, both learned the hard way:

1. **Use `val_*.json` snapshots, never `train_*.json`.** Train pools ~14,062 graphs per
   task against val's ~826 and flatters every rule (scene_kmeans: 70% on train, 56% on
   val). All shipped whitelists are cut from val.
2. **Precision here is agreement with the human list, not correctness** — that list omits
   `drawer_obj` and `cookware`, both of which VGIB found and both of which coincided with a
   rollout gain. Recall and the zero-target count are the safer signals.

**Falsified:** "hold the robot out of the cut but keep it in the subgraph." Because the
robot *is* the top of the ranking, holding it out is arithmetically just a lower threshold.
At matched size it lost to plain top-k on all 3 seeds (32/44 vs 34/44, 34/44 vs 35/44,
27/44 vs 31/44) by spending a slot on the low-scoring `OmronMobileBase`. Excluding the
robot from the subgraph outright is a different operation, and is what works.

---

## 6b. Better selection did NOT improve success (2026-09-11)

The sharpest negative in the project. `uniform_dist_weight=1e-1` (soft histogram loss,
stage-1 VGIB, seed 42, otherwise identical to `VGIB_s42_bs16`) produced a large,
size-controlled improvement in selection quality and **no change in task success**:

```
selection (val, 20 non-door tasks, shipping all_kmeans cut)
    recall                25.0%  ->  79.5%
    tasks with 0 targets   12/20 ->   1/20
    median non-robot band  0.371 ->  0.545
    matched-k recall       wins at every k from 2 to 10, peak +15.9pp at k=4
    median target rank      4.0  ->   3.0      (the loss REORDERED, not just stretched)

rollout (24 tasks, 200 scenes each)
    15.10  ->  15.06      (-0.04pp; wins 9/24, loses 8/24, nothing beyond noise)
```

**Selection recall is not a proxy for success on RoboCasa.** Any claim of the form "our
method selects the right objects, therefore it will act better" is unsupported by this
experiment and should not be made.

Two readings, not separable from this run:

(a) Selection quality genuinely does not drive success here. Consistent with §5: dropping
    `Microwave` HELPED both microwave tasks, dropping `Stove` helped TurnOffStove, and
    independent offline LOO finds objects matter only for doors and PnP — with PnP at the
    0% floor and unable to show a gain.
(b) `soft` eval hides it. Nothing is pruned in `soft`; a better ranking only reweights
    attenuation and never removes a distractor. The payoff of good selection is being able
    to PRUNE without damage, which `soft` cannot measure.

Reading (b) was tested directly by stage-2 arm A, which trains with real hard dropping.
**It is also null**, so (a) is the surviving explanation.

### Arm A settles it: the effect is null under real pruning too (2026-09-12)

`VGIB_s42_udw_s2_kmeans` — stage 2 warm-started from the udw checkpoint, trained on the
REPAIRED whitelist (79.5% recall, 1/20 blind, mean size 7.0), against the original stage-2
run on the BAD whitelist (25% recall, 12/20 blind, mean size 3.5). Both prune for real;
the whitelist is the variable.

```
                        whitelist quality            24-task    16 non-PnP
  stage 2, bad list     25.0% recall, 12/20 blind     14.56       21.72
  stage 2, good list    79.5% recall,  1/20 blind     13.92       20.81      (arm A)
  no stage 2 at all     (hard eval, untrained)        10.54       15.72
```

Arm A wins 7/24, loses 10/24, −0.65pp overall. Against a run-to-run sd of ~0.64pp on the
24-task mean (VGIB seed sd 0.45 × √2) that is **~1.0σ: no difference.**

Arm B completed the sweep on 2026-09-13. **Three points spanning 25% → 90.9% selection
quality, all identical, and not even monotone:**

```
  whitelist          recall   blind   mean size    24-task    16 non-PnP
  bad (original)      25.0%   12/20      3.5        14.56       21.72
  repaired (arm A)    79.5%    1/20      7.0        13.92       20.81
  best (arm B)        90.9%    1/20      8.0        14.31       21.28

  spread across the sweep: 0.65pp   vs run-to-run sd on the mean ~0.64pp
```

The total spread across a 3.6× change in selection quality equals one standard deviation of
run-to-run noise, and the ordering is non-monotone. **There is no trend to fit.**

**Three experiments now give the same null.** Tripling selection quality changes
nothing in `soft` eval (15.10 → 15.06) and nothing under real pruning (14.56 → 13.92).
What *does* move the number is stage-2 training itself: 10.54 → ~14 regardless of which
whitelist it distils.

**The publishable claim is the decoupling.** An object-centric policy's selected subgraph
can be brought from 25% to 79.5% agreement with human-labelled task relevance — a
reordering, not just a rescaling — with no gain in task success. Explanation quality and
task performance are separable axes on RoboCasa. That is a result about object-centric
XAI, not a failed optimisation.

Where arm A loses is concentrated on the door tasks (CloseDoubleDoor −10.0, CloseSingleDoor
−8.5), which is where a static per-task list is structurally handicapped (§5) — and note
the old stage-2 CloseDoubleDoor cell (26.5) is itself anomalously high against soft 22.5 and
SIR 14.0, so part of that −10.0 is a lucky baseline cell rather than arm A damage.

---

## 7. Currently running

| run | what it tests | status |
|---|---|---|
| `VGIB_s42_udw_s2_kmeans` (**arm A**) | stage 2 on the repaired `kmeans` whitelist. | **DONE: 13.92** (§6b) |
| `VGIB_s42_udw_s2_scenekm` (**arm B**) | stage 2 on the `scene_kmeans` whitelist. | **DONE: 14.31** (§6b) |

Nothing is queued. GPU free since 06:11 on 2026-09-13.

Both arms warm-start from `VGIB_s42_bs16_udw1e-1`, seed 42, and are otherwise byte-identical
to `VGIB_s42_bs16_stage2_lr1e-4` (14.56) — `threshold: 0.5`, `whitelist_bce_weight: 0.1`,
`vgib_noise_kl_weight: 0.0`, `use_cbm_loss: null`. `pos_weight` is re-derived per whitelist;
carrying over the old 3.69 would over-weight positives on lists twice the size and flatten
the scorer into a constant (`sparsification/*_score_std_*` is the canary).

Health check when each starts: `whitelist_match` should reach ~1.0 within the first few
epochs, as it did (1.00) on the original stage-2 run.

**Revised expectation after §6b:** near 14.5-15, not near SIR.

---

## 8. Falsified / closed

| hypothesis | verdict |
|---|---|
| Truncated KL suppresses *distinctive* objects | **False.** Aggregate corr(D, score) = −0.411, but decisive cases invert: `Sink` on TurnSinkSpout has D=0.58, the **lowest** in its graph → low D ⇒ high `p*`, so the KL pushed it **up**. The correlation comes from background clutter (`OmronMobileBase` D≈1.3–1.4, `Floor` 1.24–1.37, `Box` 1.63). |
| Per-task thresholds fix hard-drop damage | **False.** corr(keep-fraction, hard damage) = −0.002 (all 24), −0.171 (non-floor). |
| Robot held out of the cut, kept in subgraph | **False** — see §6. |
| Anneal into hard dropping during stage-1 training | **False** (student thesis §4.2.3): produced a keep-all collapse; the scorer pushes affected node scores up. The whitelist BCE is what makes hard training stable, because its target is selection-aligned and the KL is not. |
| `uniform_dist_weight` on a VGIB run | **Was a silent no-op until 2026-09-11** — the legacy-loss block was gated on `sampling_strategy != "vgib"`. Now per-term: only `uniform_except_middle` is forced off under vgib (it reads `get_num_relevant_nodes_per_task`). Any pre-fix VGIB run with this weight set measured nothing. |

---

## 9. Future work

**Ranked by expected value.**

1. **Decisive next run: SIR's recipe on a GOOD discovered list.** The one comparison that
   separates "the list" from "the recipe", and it is not yet done. What we have:

   ```
     SIR recipe + human list  (RELEVENT_NODES)      16.58
     SIR recipe + BAD discovered list (idea 6)      14.77    25% recall
     stage-2 recipe + any list (25/79.5/90.9%)      ~14.3    no trend
   ```

   Idea 6 used the 25%-recall list, so it cannot distinguish "the discovered list is worse"
   from "the stage-2 recipe is worse". Re-running idea 6 with arm B's 90.9%-recall list
   answers it: land near 16.5 and the list was the problem all along (and §6b's null is a
   property of the stage-2 recipe, not of selection in general); stay near 14.8 and the
   SIR *recipe* — `topk_PerTask` + CBM + histogram loss, with per-episode language on the
   doors — is the whole 1.95pp, independently of which nodes are selected.
   Cost ~17h; it is the highest-value use of the now-idle GPU.
2. **Stop using selection recall as a success proxy.** Every "better explanation" claim in
   this project has been validated against `RELEVENT_NODES` agreement, never against
   rollouts. §6b is the first direct test and it came back null.
3. **Hybrid list** — seed from `RELEVENT_NODES`, add VGIB's highest-ranked additions, keep
   SIR's language-resolved door handling. Motivated by §5: VGIB contributes `drawer_obj`
   and `cookware`; the human list contributes the objects VGIB's cut loses. Neither alone.
4. **Replicate seeds 43/44** for stage 2 / hard / noise. Every decomposition number in §4
   is single-seed.
5. **Language grounding for the door tasks** — any per-task list is structurally capped
   here. Needed before claiming an unsupervised method matches SIR overall.
6. **Charge robot nodes extra KL** rather than exempting them (the original "idea 2" had
   the sign backwards: exempting removes down-pressure and pushes them *higher*).
7. **PnP is floored at ~0%** on every method — the 8 PnP tasks average 0.17% (SIR, best
   single task 0.67%) and 0.15% (VGIB). They contribute no signal but a third of the task
   count, compressing every 24-task mean toward zero. Consider reporting the **16 non-PnP
   tasks** as the primary table — SIR 24.81, VGIB soft 21.90, Image 19.08 — which preserves
   the ordering while making the effect sizes legible, and raise the PnP floor separately.

---

## 10. Reproduction

```bash
# stage-1 VGIB
python main.py manager.task_names=[ALL] manager.batch_size=16 \
  trainer.train_bool=true trainer.test_bool=false trainer.seed=42 \
  method/graph_encoder=xai_gnn_vgib

# cut a whitelist from the VAL snapshot (never train)
python -m utils.select_subgraph \
  logs/robocasa_vgib/VGIB_s42_bs16/node_scores/val_epoch049.json \
  --scene-kmeans -o .../subgraph_scene_kmeans.yaml

# stage 2 on that whitelist
python main.py ... method/graph_encoder=xai_gnn_vgib_stage2 \
  method.graph_encoder.sparsification_layer.node_whitelist=<yaml> \
  method.graph_encoder.sparsification_layer.whitelist_pos_weight=<neg/pos> \
  trainer.init_from=logs/robocasa_vgib/VGIB_s42_bs16

# eval (200 unique scenes/task)
python main.py manager.task_names=[ALL] trainer.train_bool=false trainer.test_bool=true \
  trainer.eval_n_times=200 manager.times_repeat=1 trainer.eval_checkpoint=_final \
  manager.parallel_eval=true manager.num_envs=10 trainer.model_path=<run dir>
```

Diagnostics: `utils/diagnose_selection_rules.py` (rule scoring, val by default),
`utils/diagnose_robot_holdout.py` (the falsified holdout, kept as a record).

---

---

## 11. Evaluation protocol: is RoboCasa in- or out-of-distribution?

**In-distribution.** `create_kitchen_env` builds the eval env from the demo file's own
`env_args`, inheriting every scene-generating parameter unchanged — `layout_ids=-1`, the
same 10 `style_ids`, `obj_instance_split='A'`, `generative_textures='100p'`,
`randomize_cameras=True`. Each rollout calls `env.reset()` under a deterministic
per-rollout seed. Same generator, fresh draws.

Scenes being visually novel is not evidence of shift: object positions, fixture placement
and camera pose are sampled continuously, so every scene differs from every demo. That is
what sampling a continuous distribution does. **Distribution shift is a property of the
sampling distribution, not of individual samples** — do not describe this setup as OOD.

What can be claimed, measured on 20 scenes per task at the real eval seeds (`n=480`):

| task | demo (layout,style) pairs | unseen pair | unseen object instances | exact scene repeats |
|---|---|---|---|---|
| CloseDrawer | 45/100 | 50% | 62% | 0/20 |
| OpenDrawer | 43/100 | 50% | 63% | 0/20 |
| CloseDoubleDoor | 39/100 | 75% | 72% | 0/20 |
| CloseSingleDoor | 44/100 | 35% | 74% | 0/20 |
| OpenDoubleDoor | 39/100 | 70% | 67% | 0/20 |
| OpenSingleDoor | 42/100 | 55% | 79% | 0/20 |
| TurnOffMicrowave | 41/100 | 80% | 57% | 0/20 |
| TurnOnMicrowave | 40/100 | 60% | 48% | 0/20 |
| TurnOnSinkFaucet | 42/100 | 70% | 76% | 0/20 |
| TurnOffSinkFaucet | 41/100 | 70% | 63% | 0/20 |
| TurnSinkSpout | 38/100 | 60% | 61% | 0/20 |
| TurnOnStove | 39/100 | 65% | 30% | 1/20 |
| TurnOffStove | 38/100 | 35% | 20% | 0/20 |
| CoffeePressButton | 46/100 | 65% | 0% | 3/20 |
| CoffeeServeMug | 40/100 | 70% | 10% | 1/20 |
| CoffeeSetupMug | 42/100 | 60% | 0% | 2/20 |
| PnPCabToCounter | 40/100 | 85% | 67% | 0/20 |
| PnPCounterToCab | 44/100 | 65% | 77% | 0/20 |
| PnPCounterToMicrowave | 40/100 | 45% | 29% | 0/20 |
| PnPCounterToSink | 45/100 | 55% | 67% | 0/20 |
| PnPCounterToStove | 44/100 | 35% | 22% | 0/20 |
| PnPMicrowaveToCounter | 39/100 | 65% | 28% | 0/20 |
| PnPSinkToCounter | 38/100 | 65% | 53% | 0/20 |
| PnPStoveToCounter | 43/100 | 50% | 28% | 0/20 |
| **ALL 24** | — | **60%** | **53%** | **7/480** |

So: **60% of eval scenes use a (layout, style) pair that task never demonstrated, 53% of
object instances were never demonstrated, and 7 of 480 scenes reproduced a demo's
(layout, style, asset-set) — and even those differ in continuous object placement.**

Per-task object novelty tracks how many objects the task samples. The three Coffee tasks
and TurnOnStove sample a single object from a small category pool (`mug`, `pot`), so their
instance novelty is 0-30% and all 7 exact repeats fall there. Tasks sampling 3-4 objects
run 60-80% novel.

### Two limits on stronger claims

1. **Held-out layout/style is not available.** Pooled over all 24 tasks' 1293 demos, **all
   100** (layout, style) pairs appear in training. Novelty is per-task only; the
   multi-task policy has seen every kitchen configuration somewhere.
2. **`obj_instance_split='B'` is not trustworthy on this install.** The threshold is
   `max(len(choices) - 3, ceil(len(reg_choices)/2))` where `choices` is a dict keyed by
   *registry* — always length 1 — so the documented "all but the last 3 instances" rule
   never fires and it is a plain half/half cut on registry list order. That order differs
   between our robocasa and the one that collected the demos: `fork_5`, `pear_19` and
   `potato_5` land in B locally but are tagged `"split": "A"` in the demo metadata. Our B
   is not the complement of their A. (All 616 distinct demo assets ARE present locally, so
   nothing about training is compromised — the drift is ordering, not missing assets.)

### Cameras are randomized

`randomize_cameras=True`, inherited at eval. Both fixed cameras the model consumes are
re-posed every episode; only the wrist camera is pose-stable:

```
robot0_agentview_left    50 distinct poses / 54 demos   pos sd ~0.05 m, range 0.22-0.29 m
robot0_agentview_right   50 distinct poses / 54 demos   pos sd ~0.05 m, range 0.18-0.28 m
robot0_eye_in_hand        1 distinct config             (wrist-mounted)
```

Relevant to any inhand-vs-fixed-camera claim: the fixed views carry per-episode pose jitter
that the wrist view does not.

Tooling: `utils/scene_overlap_all.py` (the table above), `utils/diagnose_scene_overlap.py`
(one task, verbose). Both run with the renderer off and need no GPU.

## 12. Per-task results

| task | SIR (3 seeds) | VGIB soft (3 seeds) | Image (3 seeds) | VGIB noise s42 | VGIB hard s42 | VGIB stage2 s42 | SIR@VGIBlist s42 |
|---|---|---|---|---|---|---|---|
| CloseDoubleDoor | 16.5 ± 2.8 | 20.0 ± 2.2 | 33.5 ± 5.6 | 16.0 | 13.5 | 26.5 | 13.5 |
| CloseDrawer | 69.3 ± 1.2 | 74.0 ± 1.0 | 71.3 ± 3.2 | 65.0 | 66.5 | 76.0 | 73.0 |
| CloseSingleDoor | 37.7 ± 2.3 | 40.5 ± 3.8 | 48.8 ± 7.5 | 38.0 | 38.0 | 39.5 | 37.0 |
| CoffeePressButton | 14.3 ± 1.2 | 7.7 ± 1.6 | 8.8 ± 3.2 | 8.0 | 0.5 | 5.5 | 9.0 |
| CoffeeServeMug | 12.7 ± 1.3 | 6.2 ± 2.0 | 3.8 ± 0.8 | 6.5 | 8.5 | 10.0 | 7.5 |
| CoffeeSetupMug | 0.3 ± 0.3 | 0.3 ± 0.3 | 0.2 ± 0.3 | 0.0 | 0.0 | 0.5 | 0.0 |
| OpenDoubleDoor | 3.5 ± 1.5 | 3.3 ± 1.6 | 2.5 ± 1.0 | 1.5 | 1.0 | 2.0 | 1.5 |
| OpenDrawer | 12.3 ± 3.3 | 8.3 ± 2.4 | 10.5 ± 1.0 | 8.0 | 4.0 | 9.0 | 11.5 |
| OpenSingleDoor | 14.7 ± 1.0 | 10.5 ± 3.3 | 9.0 ± 1.3 | 10.5 | 6.5 | 8.0 | 10.5 |
| PnPCabToCounter | 0.2 ± 0.3 | 0.2 ± 0.3 | 1.7 ± 0.3 | 1.0 | 0.5 | 0.5 | 0.0 |
| PnPCounterToCab | 0.3 ± 0.3 | 0.2 ± 0.3 | 0.3 ± 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| PnPCounterToMicrowave | 0.0 ± 0.0 | 0.2 ± 0.3 | 0.2 ± 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| PnPCounterToSink | 0.7 ± 0.3 | 0.5 ± 0.0 | 1.0 ± 0.5 | 1.0 | 1.0 | 1.5 | 0.5 |
| PnPCounterToStove | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PnPMicrowaveToCounter | 0.0 ± 0.0 | 0.2 ± 0.3 | 0.3 ± 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| PnPSinkToCounter | 0.2 ± 0.3 | 0.0 ± 0.0 | 0.7 ± 0.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| PnPStoveToCounter | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.8 ± 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| TurnOffMicrowave | 24.8 ± 3.0 | 28.0 ± 3.6 | 30.3 ± 0.8 | 25.0 | 5.5 | 26.5 | 29.5 |
| TurnOffSinkFaucet | 58.5 ± 2.5 | 49.0 ± 3.1 | 13.7 ± 3.8 | 45.5 | 20.5 | 44.5 | 46.5 |
| TurnOffStove | 9.5 ± 2.2 | 7.7 ± 1.3 | 3.5 ± 1.3 | 7.5 | 7.0 | 7.0 | 9.5 |
| TurnOnMicrowave | 13.5 ± 2.0 | 11.8 ± 5.4 | 16.3 ± 7.2 | 9.5 | 4.5 | 14.5 | 16.0 |
| TurnOnSinkFaucet | 28.3 ± 0.8 | 20.2 ± 4.5 | 14.3 ± 1.6 | 16.5 | 13.0 | 16.5 | 16.5 |
| TurnOnStove | 24.8 ± 3.3 | 20.5 ± 1.7 | 10.0 ± 1.3 | 25.0 | 17.5 | 15.0 | 22.5 |
| TurnSinkSpout | 56.2 ± 1.9 | 42.3 ± 2.0 | 28.7 ± 1.8 | 43.5 | 45.0 | 46.5 | 50.0 |
| **MEAN (24)** | **16.60 ± 0.19** | **14.65 ± 0.45** | **12.93 ± 0.17** | **13.67** | **10.54** | **14.56** | **14.77** |

Seeded columns are mean ± sd over seeds 42/43/44; single-seed columns are seed 42 only.
All cells are 200 rollouts over 200 unique scenes.
