# SIR: Structured Image Representations for Explainable Robot Learning

## Goal of the Repository

This repository accompanies the paper **"SIR: Structured Image Representations for Explainable Robot Learning"**. Its goal is to train and evaluate a robot-manipulation policy that represents visual scenes as **object-centric graphs** rather than (or in addition to) raw images. Moving from flat image embeddings to explicit per-object nodes is intended to:

1. **Improve explainability** — each node corresponds to a named object, so the policy's inputs can be inspected and the relevant objects for a given task can be identified.
2. **Enable structured modality fusion** — per-object features (one-hot labels, language embeddings, bounding boxes, cropped image features, 3D point geometry) are combined on a shared graph.
3. **Support sparsification / XAI** — a dedicated module can learn which nodes matter most per task, reducing the graph to a task-relevant subset and making the selection inspectable at rollout time.

The policies are trained with behavior cloning on RoboCasa kitchen demonstrations and evaluated in the corresponding simulator.

## High-Level Architecture

The pipeline is configured with **Hydra** and logged with **Weights & Biases**. A single run is orchestrated by [main.py](main.py) → [trainer.py](trainer.py) and composes three pluggable components:

| Component | Purpose | Implementation |
|---|---|---|
| **Manager** | Owns dataset + environment for a task family | [manager/robocasa_manager.py](manager/robocasa_manager.py) |
| **Method** | Learning algorithm (loss, optimizer, inference) | [method/diffusion.py](method/diffusion.py) |
| **Networks** | Encoders + action generator | [networks/](networks/) |

Each of the three encoders is optional and activated simply by listing modalities in the manager config:

- **Proprioceptive**: MLP over robot state vectors (`prop_modalities`)
- **Image**: ResNet / EfficientNet / ViT optionally with FiLM language conditioning, in [networks/vision_encoder/cnn.py](networks/vision_encoder/cnn.py) (`image_modalities`; `inhand` / `left` / `right`, all 128×128)
- **Graph**: multi-modality GATv2 GNN in [networks/graph_encoder/gnn.py](networks/graph_encoder/gnn.py) (`graph_modalities`)

The action policy is a **Multimodal Diffusion Transformer (MDT)** defined in [networks/action_generator/mdt.py](networks/action_generator/mdt.py). A transformer encoder fuses the goal token (CLIP text embedding) with the observation tokens; a FiLM-conditioned transformer decoder denoises the action sequence conditioned on that context and the diffusion sigma. Training uses Karras-style preconditioning and log-logistic sigma sampling; inference uses DDIM sampling ([method/utils.py](method/utils.py)).

### Graph readout: pooled vs. per-node

`method.graph_encoder.pool_name` decides what the graph contributes to the MDT context:

- `mean` / `add` / `max` — one pooled embedding per modality, i.e. **one token** for the whole scene. Under mean pooling with a sparsifier, the normalisation divides by the count of nodes with `probs > 0`, so dropped nodes do not dilute the pool.
- `none` — **one token per object node** plus a validity mask, so the MDT cross-attends to each object separately and the language goal can route to a specific node. Requires `method.action_generator.max_graph_nodes >= ` the largest graph (default 32); the MDT sizes its position embedding for `max(obs_window, max_graph_nodes)` tokens per modality, and `cross_attn_mask` excludes the padded slots.

[trainer.py](trainer.py) `set_runtime_parameter` forces `max_graph_nodes = obs_window` whenever pooling is *not* `none`, so pooled checkpoints keep their historical position-embedding shape.

## The "Structured Image Representation"

The core contribution lives in [utils/generate_graph_dataset_robocasa.py](utils/generate_graph_dataset_robocasa.py), [utils/create_graphs.py](utils/create_graphs.py), and [dataloader/utils.py](dataloader/utils.py).

For each demonstration step, the **ground-truth segmentation masks** (MuJoCo `*_segmentation_class`, taken from `dataset_raw.hdf5` offline and read live off the simulator at rollout) from the **left** and **right** agentview cameras are used to:

1. Detect the set of object classes present (list in `OBJECT_NAMES_IMAGES`).
2. Build a node per object, featurized by one or more of the following **graph modalities**:
   - `one_hot_labels` — class identity
   - `bb_coordinates` — 10-dim normalized bounding-box coordinates
   - `cropped_image_feature` — embedding from a pretrained `SimpleImageEncoder` / ResNet18 over the cropped object patch (see [trained_models/cropped_image_feature/](trained_models/cropped_image_feature/))
   - `siglip_text_emb` — 768-dim SigLIP text embedding of `"The current observed object in the image is {name}"`
   - `point_geom` — 17-dim metric 3D descriptor reconstructed from depth + segmentation ([utils/point_geom.py](utils/point_geom.py)): centroid in the robot-base frame, PCA orientation as two rotation-matrix columns, per-axis extents, centroid−gripper offset and distance, and a visibility scalar. View-invariant (built from both views at once), so unlike the other spatial modalities it is *not* doubled by fusion. It cannot be produced by `create_graphs_and_save` — it needs a depth replay via `python -m utils.generate_point_geom_dataset --task <T>`, and rollouts need `camera_depths` on, which the manager enables automatically when `point_geom` is in `graph_modalities`.
3. Add a fully-connected, bidirectional edge set (weight 1) between every pair of nodes.

Left and right graphs are either kept as two parallel graphs (`use_graph_fusion=False`) or **fused into one graph per step** (`use_graph_fusion=True`). Fusion ([dataloader/utils.py](dataloader/utils.py) `fuse_graphs`) takes the union of node names across views and combines features per modality: identity-like features (one-hot, SigLIP) are shared; spatial features (bounding boxes, cropped features) are concatenated from both views — or left-view only when a jointly pre-trained fusion encoder is used.

Multiple modalities can additionally be concatenated along the feature dimension (`use_splitted_modalities=False`) to form a single richer node feature, or kept separate so that each modality is encoded by its own GNN branch.

**Relevance ground truth.** `RELEVENT_NODES` in [utils/generate_graph_dataset_robocasa.py](utils/generate_graph_dataset_robocasa.py) maps each task to its hand-written relevant object set; `get_num_relevant_nodes_per_task` derives the per-task top-k budget (door tasks add +1 for the door named in the language goal). `NODE_ALIASES` maps `Stove → {Stovetop}`, because the two segmentation classes are complementary across demos (~56%/44%) and without the alias the real target was scored as irrelevant in the `Stovetop` demos. Aliasing here rather than widening `RELEVENT_NODES` leaves the top-k budget unchanged.

## Explainability / Sparsification

[networks/graph_encoder/gnn.py](networks/graph_encoder/gnn.py) provides two graph encoder variants:

- **`Multi_GNN`** (configured via [configs/method/graph_encoder/gnn.yaml](configs/method/graph_encoder/gnn.yaml)) — a plain multi-modality GATv2 stack with no sparsification.
- **`Multi_XAI_GNN`** — wraps each GNN branch with a `Sparsification_Module` that scores nodes and reduces the graph before / around the downstream GNN. One config per sampling strategy (table below).

### Sparsification_Module

A `TransformerCoarsening` scorer (FiLM-conditioned `TransformerFiLMDecoder`) assigns a relevance score per node conditioned on the CLIP language embedding. The scorer is **self-attention only** — `use_cross_attention=False`, since `forward()` never passes a context and a cross-attention block there would allocate parameters that never receive a gradient. Language enters through the FiLM conditioning instead. Scores pass through a sigmoid, then one of five **sampling strategies** decides what survives:

| Strategy | Config | Selection rule | Supervision |
|---|---|---|---|
| `topk_PerTask` | [xai_gnn.yaml](configs/method/graph_encoder/xai_gnn.yaml) | exactly `k = get_num_relevant_nodes_per_task(task)` nodes per graph | CBM loss on `RELEVENT_NODES` — **this pair is the SIR method** |
| `topk` | [xai_gnn_topk.yaml](configs/method/graph_encoder/xai_gnn_topk.yaml) | top `sample_abs` nodes (default 5), task-agnostic | none |
| `threshold` | [xai_gnn_threshold.yaml](configs/method/graph_encoder/xai_gnn_threshold.yaml) | `probs > threshold` (default 0.5); kept count varies per graph | none |
| `vgib` | [xai_gnn_vgib.yaml](configs/method/graph_encoder/xai_gnn_vgib.yaml) | no pruning during training; at eval, `eval_prune_mode` + `threshold` | compression KL only (unsupervised) |
| `fixed` | [xai_gnn_vgib_stage2.yaml](configs/method/graph_encoder/xai_gnn_vgib_stage2.yaml) (alternative mode) | a `node_whitelist` **is** the mask; the scorer never runs | n/a |

`topk_PerTask`, `topk` and `threshold` apply the same straight-through estimator (`indexes = ((kept_mask) - probs).detach() + probs; probs = probs * indexes`) so gradients flow back into the scorer. Edge weights are modulated as `probs[src] * probs[dst]` and consumed multiplicatively by `WeightedGATv2Conv` (a custom subclass of `GATv2Conv` that multiplies attention scores by `edge_weight` rather than treating it additively).

**Multiplicative pooling** (`multiplicative_pooling: true`): after the GNN computes node features, each feature vector is scaled by the node's sparsification probability (`output * probs`) before pooling. Under `pool_name: none`, nodes the sparsifier zeroed are additionally removed from the validity mask so they are not attendable as all-zero tokens. This closes the gradient path from the readout back through the sparsification scores.

Auxiliary losses are computed every forward pass (training + validation) and aggregated in `Multi_XAI_GNN.aux_losses`, keyed `"{modality}/{loss_name}"`. During training they contribute to `total_loss` in [method/diffusion.py](method/diffusion.py); during validation they are logged for monitoring only.

**Score-shaping regularizers** (all computed every forward pass in the non-VGIB strategies; only `uniform_dist_weight` is active by default):
- `variance_weight` — negative variance of scores; penalizes low spread, pushing scores toward 0 or 1
- `sum_mask_weight` — absolute deviation from a target total probability mass
- `entropy_reg_weight` — `sum(p·log(p))`; naturally negative, keeps scores from collapsing
- `uniform_dist_weight` — soft-histogram MSE vs. uniform distribution (default active: `1e-1`)
- `uniform_except_middle_weight` — bimodal variant: K scores pushed high, N-K scores pushed low
- `threshold_sum_weight` — mass above the cutoff; computed in `threshold` and `vgib` modes, default 0.0

The whole per-graph regularizer loop is **skipped under `vgib`**. That is not only an optimisation: `uniform_except_middle` sizes itself from `get_num_relevant_nodes_per_task`, so running it would put the hand-written relevance table back into a path whose entire claim is that it does not use one.

**CBM (Concept Bottleneck Model) loss** (`use_cbm_loss: "nodes_att"`, weight `lcbm_weight: 1e-1`): direct supervision on the scorer using `RELEVENT_NODES`. Formula: `(-probs[relevant].sum() + probs[~relevant].sum()) / N`. Computed only during training. Requires the raw language goal text (stored in `item['goal']['lang_text']` by [dataloader/datasets.py](dataloader/datasets.py)) for door tasks where the specific door is extracted from the instruction. **Only applicable to `topk_PerTask`** — every other strategy is task-agnostic or unsupervised and sets `use_cbm_loss: null`.

**Active losses per sampling strategy** (non-zero weights only):

| Loss | `topk_PerTask` | `topk` | `threshold` | `vgib` | stage 2 |
|---|:---:|:---:|:---:|:---:|:---:|
| `coars_uniform_loss` | 0.1 | 0.1 | 0.1 | — | — |
| CBM `nodes_att` | 0.1 | — | — | — (must stay off) | — (must stay off) |
| `coars_threshold_sum_loss` | — | — | 0.0 | 0.0 | 0.0 |
| `vgib_noise_kl_loss` | — | — | — | 1e-2 | 0.0 |
| `whitelist_bce_loss` | — | — | — | — | 0.1 |

## VGIB: unsupervised sparsification

`sampling_strategy: vgib` implements the Variational Graph Information Bottleneck (Yu et al., CVPR 2022) as the sparsifier. The difference from `xai_gnn` is not the loss *shape* but what supervises the selection: `use_cbm_loss` is `null` and nothing in the VGIB path reads `RELEVENT_NODES`, so pressure to drop a node comes only from the compression KL and pressure to keep one only from the prediction loss. An explanation produced this way is evidence *about* the relevance table rather than a restatement of it — which is exactly why `use_cbm_loss` must stay null, and why the regularizer loop is skipped.

### Training: discard by dissolving into noise

Nothing is pruned during training. Per graph, `_vgib` computes the node-wise mean `mu_G` and std `sigma_G` over that graph's nodes, then replaces node *v*'s features with a draw from

```
z_v ~ N( p_v·x_v + (1-p_v)·mu_G ,  ((1-p_v)·sigma_G)^2 )
```

so a node is discarded by being *dissolved into its graph's own noise*, which is what makes the discard differentiable. The noised features are written back onto `batch.x` and the GNN runs on them — otherwise the compression term would be computed against a corruption the policy never sees.

The compression term is the closed-form KL between that noisy node distribution and the noise prior `N(mu_G, sigma_G^2)`, selected by `vgib_noise_kl_mode`:

- **`truncated`** (default) — the quadratic part only. Each node has a stable interior optimum at `p* = 1/(1 + d²)`, where `d` is how far its features sit from the graph mean, so the score becomes a *readout of distinctiveness*: graded, rankable, and still there to be thresholded afterwards. Raising `vgib_noise_kl_weight` here pulls scores harder toward `p*`, an interior point — it does **not** evict more, which is why `1e-2` is stable in this mode.
- **`full`** — the complete Gaussian KL, restoring the `-log(1-p) - ½` term. More principled as a bound and wrong here: its optimum is `p → 0` for *every* node regardless of `d`, so nothing survives except what the prediction loss actively holds up — and that loss is nearly blind to sparsification (measured: val MSE moved 0.058 → 0.078 while rollout success moved 0.720 → 0.185). At `1e-3` it evicted everything but `PandaMobile`; at `1e-4` it left a shapeless 0.36–0.67 band with no gap to cut at.
- **`eq10`** — the paper's graph-level bound (Prop. 4.1): one KL between the pooled noisy representation and the pooled prior rather than a per-node sum. `-log(A)` acts as a barrier against keeping everything whose pressure decays as ~1/A once eviction starts. Different scale entirely; needs its own weight sweep.

`threshold_sum_weight` adds linear rent on scores above `threshold` (0.0 = plain VGIB). Raise it (~1e-2) if the scorer learns to park everything just above the bar, which costs nothing once the KL is paid and makes `threshold` meaningless.

### Evaluation: `eval_prune_mode`

VGIB has no k budget, so `threshold` alone decides subgraph size — and it is not guessable in advance. Train once, then read the cut off the score distribution (see *Node scores* below).

- **`soft`** (default) — no threshold at all; every node is attenuated toward the graph mean in proportion to `(1 - p)`, using the expectation `p·x + (1-p)·mu` the way eval-time dropout uses its expectation. The only mode whose number reflects the *scores alone* rather than the scores plus a threshold picked afterwards, so this is the right setting for a KL-weight sweep.
- **`noise`** — threshold, but sub-threshold nodes are redrawn from the graph's Gaussian instead of being removed. The graph keeps its size and topology; the dropped nodes carry no information.
- **`hard`** — threshold, zero the score, cut the edges (`edge_weight *= probs[src]·probs[dst]`). This is the deterministic sparse subgraph you would ship as the explanation, and also the furthest from anything training showed the policy: on the first sweep it cost CloseDrawer 0.600 → 0.185 on the very same checkpoint, purely from the cut. Report it; do not tune on it.

`eval_prune_mode` is read only inside the `vgib` branch and is inert under the other strategies. Because `threshold_sum_weight` defaults to 0.0, `threshold` never enters a gradient under `truncated` — it is a **pure eval-time knob**, so the cut can be chosen and re-chosen after training without retraining anything.

### Stage 2: burning the discovered subgraph into the scorer

Stage 1 scores nodes softly, so the selection flickers from timestep to timestep. [xai_gnn_vgib_stage2.yaml](configs/method/graph_encoder/xai_gnn_vgib_stage2.yaml) fixes that:

```bash
python -m utils.select_subgraph <stage1>/node_scores/val_epoch049.json --threshold 0.6 -o subgraph.yaml
python main.py method/graph_encoder=xai_gnn_vgib_stage2 \
  method.graph_encoder.sparsification_layer.node_whitelist=/abs/path/subgraph.yaml \
  trainer.init_from=/abs/path/to/stage1_run_dir
```

The scorer still runs and its own `probs > threshold` still decides, but a `pos_weight`-corrected BCE on the **logits** pulls scores apart onto the subgraph stage 1 chose, so selection becomes stable and the thing running at inference is a scorer rather than a lookup table. `sampling_strategy: fixed` is the alternative: the whitelist becomes the mask outright and the scorer never runs — simpler, but nothing is learnable at inference.

Two calibration notes carried in the config:

- `train_diffusion_loss` here is ~0.055, **not** O(1). A BCE weight eyeballed for "loss ≈ 1" is ~20× too big: at 10.0 the BCE was 99.3% of `total_loss` and flattened the scorer into an input-independent constant ≈0.6 for every node by epoch 10. `sparsification/{train,val}_score_std_*` is the canary — it collapses toward 0 an epoch or two before the means visibly move.
- Graphs carry 18–27 node types while a whitelist keeps 1–5 (~19% positives), so unweighted BCE is mostly "push everything down" and a constant output is attractive. Hence `whitelist_pos_weight: 4.3`.

`sparsification/{train,val}_whitelist_match_{mod}` reports the fraction of nodes whose hard selection agrees with the whitelist; it should reach ~1.0 within the first stage-2 epochs, and if it does not the BCE weight is wrong.

**The whitelist must come from stage 1's own node scores, never from `RELEVENT_NODES`.** That is the whole difference between distilling an unsupervised discovery and importing the answer key. `use_cbm_loss` stays `null` in stage 2 for the same reason.

## Node scores: the durable record

A sparsifier with no k budget cannot be thresholded without knowing what the scores look like, so the **pre-mask** sigmoid scores (`raw_probs`, cloned before any selection zeroes them) travel out of `Multi_XAI_GNN.forward` in `last_sparsification` alongside node names and per-graph task names.

[utils/node_scores.py](utils/node_scores.py) `NodeScoreTracker` accumulates them keyed by `(modality, task, node name)`. Three call sites feed it, so all three splits produce directly comparable snapshots:

- [trainer.py](trainer.py) — one tracker per train epoch and per val epoch (the val one under the EMA shadow, i.e. the weights a threshold would actually be applied to)
- [envs/robocasa/kitchen.py](envs/robocasa/kitchen.py) — the sequential tester
- [envs/robocasa/parallel/policy_server.py](envs/robocasa/parallel/policy_server.py) — the parallel path; the server owns every forward pass, so it is the one place that sees a whole evaluation

Outputs:

- `<working_dir>/node_scores/{train,val}_epoch<NNN>.json` and `.../eval_{final|epoch<e>}_round<i>.json` — per-modality, per-task, per-node `mean` / `std` / `count` plus `graphs_seen`. This is the durable record that every tool below reads.
- wandb metrics `node_scores/<split>/<task>/<object>` — plain scalars, one per (task, object), pooled across modalities, plotted against their own `node_epoch` x-axis. Deliberately *not* a `wandb.plot.line_series`: a custom chart renders as a static image you cannot hover, zoom, smooth, or compare across runs. These are ordinary metrics instead, so an object's score behaves like a loss curve. They are excluded from `main.log` — ~80 lines an epoch would bury everything else.

### Tooling around the snapshots

| Tool | What it does |
|---|---|
| [utils/suggest_threshold.py](utils/suggest_threshold.py) | Per (modality, task), ranks nodes by mean score and proposes two cutoffs: **gap** (midpoint of the largest gap between consecutive means — the one that matters, since a clean gap means the scorer committed) and **k-split** (midpoint at `get_num_relevant_nodes_per_task`, a reference line only). Marks `RELEVENT_NODES` membership with `*` — diagnostic only, nothing feeds back into training. Ends with a paste-ready Hydra override and warns when the per-task cutoffs span >0.2, which means one global threshold is the wrong tool. |
| [utils/select_subgraph.py](utils/select_subgraph.py) | Cuts a snapshot into the stage-2 whitelist YAML. Prints the ranked table per task, a keep-count scan over cutoffs 0.3…0.8, and flags nodes within one std of the cut (stage-1 flicker risk), empty/1-node subgraphs, and dropped `RELEVENT_NODES`. Run it *without* `--threshold` first: tables only, nothing written. `--task-threshold Task=0.8` is repeatable, at the cost of size-matched cross-task comparison. |
| [utils/compare_node_scores.py](utils/compare_node_scores.py) | Cross-round consistency of the eval snapshots (pairwise Spearman, top-k Jaccard, per-node CV), eval-vs-train/val agreement, names present on only one side (live-segmentation vs. dataset vocabulary drift), and precision@k of the merged eval top-k against `RELEVENT_NODES`. |
| [utils/backfill_node_scores.py](utils/backfill_node_scores.py) | Replays on-disk snapshots into an existing wandb run, for runs that logged the wrong thing or nothing. Uses the `node_epoch` axis because a resumed run continues from its last global step and replayed steps would be dropped as out-of-order. |
| [utils/make_wandb_panels.py](utils/make_wandb_panels.py) | Builds one wandb line panel per `(split, task)` so a task's objects share axes instead of arriving as ~80 tiny charts. **Run from a separate venv** — `wandb-workspaces` pulls a newer `wandb` with it, and upgrading that under a running evaluation corrupts a comparison. |

## Eval-only ablations and warm starts

Read from `trainer.*` in **this** config rather than from the checkpoint, so a trained model can be re-examined without retraining:

- `trainer.graph_eval_mode` — `none` (normal) · `full` (keep every node, the in-distribution upper bound) · `random` (a random subgraph of the *same size* the model selected) · `zero` (blank the graph embedding at token level, out of distribution). `random` is the control that separates *"the policy needs this subgraph"* from *"the policy needs a subgraph of this size"* — without it, a high score on a 3-node explanation says nothing about *which* 3 nodes. `full`/`random` act inside the sparsifier via `Multi_XAI_GNN.set_graph_swap`; `zero` acts on the MDT token in `Diffusion.preprocess_batch`. A swap always yields a hard 0/1 subgraph with its edges cut, so it must be compared against `eval_prune_mode=hard` — under `soft`/`noise` the `graph_eval_mode=none` reference is *not* pruned and the pair would differ by the pruning as well as by the node choice. `set_graph_swap` warns when that mismatch is set up.
- `trainer.task_thresholds` — per-task eval cutoff, e.g. `{CloseDrawer: 0.878, CoffeePressButton: 0.797}`; unlisted tasks fall back to the encoder's scalar `threshold`. The gap in the score distribution sits in a different place per task, so one global bar necessarily over-prunes one and under-prunes another. Training always uses the scalar, since the bar the loss was calibrated on has to be the one it trained against.
- `trainer.init_from` / `init_from_suffix` / `init_from_ema` — warm-start a **training** run from another run's weights (stage 2 points this at the stage-1 run dir). Weights only: this run keeps its own config, optimizer and LR schedule, which is exactly what stage 2 needs (its encoder config must stay `threshold` + whitelist BCE rather than reverting to `vgib`). With `init_from_ema` the source's EMA shadow is copied into the live parameters before this run re-registers its own EMA from them, so training continues from the weights whose node scores were thresholded. **Must stay `null` for stage 1**, or every run silently continues someone else's checkpoint. Ignored when `train_bool=false`; evaluation uses `model_path`.

Two further optional trainer knobs are read with defaults and are absent from `main.yaml`: `trainer.ema_decay` (default `0.999`) and `trainer.grad_clip` (default `0.0` = off).

### Re-evaluating a checkpoint

With `train_bool=false`, `check_if_model_is_loaded` replaces the `manager` and `method` configs wholesale with the checkpoint's own copy, then **re-applies any explicit `manager.*` / `method.*` command-line override on top** — so architecture comes from the checkpoint while eval-time knobs (`num_envs`, `parallel_eval`, `eval_prune_mode`, …) come from the command line. Two consequences worth knowing:

The VGIB and stage-2 keys (`eval_prune_mode`, `vgib_noise_kl_*`, `node_whitelist`, `whitelist_*`) are listed inert in [xai_gnn.yaml](configs/method/graph_encoder/xai_gnn.yaml) — the config the defaults list composes — because Hydra rejects an override of a key absent from the composed struct. Keeping the default a superset is what lets `method.graph_encoder.sparsification_layer.eval_prune_mode=hard` work on any checkpoint without also re-specifying the encoder group.

The run name is likewise derived from the **checkpoint's** manager/method (`naming_config` in [main.py](main.py)), so a re-evaluated VGIB checkpoint logs as `VGIB_s42` rather than as whatever the defaults list composed. An explicit `wandb.run_name=` still wins, and is worth setting when several evals of one checkpoint would otherwise share a name.

## Training Metrics

`Multi_XAI_GNN.forward` stashes per-modality state in `last_sparsification` after each forward pass. [trainer.py](trainer.py) reads these at the end of each epoch and logs to wandb:
- `sparsification/{train,val}_kept_nodes_{mod}` — fraction of nodes surviving selection (from the last batch of the epoch)
- `sparsification/{train,val}_kept_edges_{mod}` — fraction of edges with non-zero weight after sparsification
- `sparsification/{train,val}_whitelist_match_{mod}` — stage 2 only

All coarsening losses are logged with `train_spars/` and `val_spars/` prefixes. The CBM loss (`train_spars/{mod}/cbm_loss`) appears only during training since it requires gradient computation.

Under `vgib`, `kept_*` ratios are reported against the threshold cut in **both** phases: during training `probs` is soft and never exactly zero, so `probs != 0` would report every node as kept and the logged sparsity would be a flat 1.0 all run. Under `soft` eval the edges are never gated, so `kept_edges` stays 1.0 by construction.

## Rollout Sparsification Visualization

After each evaluation rollout, the tester reads `method.graph_encoder.last_sparsification` (stashed at every inference step) to accumulate:

- **`presence_counts`** — how many steps each node survived selection (aggregated over the rollout).
- **`selection_timeline`** — which node names were selected at each individual step.

One PNG per modality per rollout is saved via `visualize_sparsification_rollout` ([envs/robocasa/utils.py](envs/robocasa/utils.py)) under

```
<working_dir>/sparsification/<task>/repeat_<test_count>/
<working_dir>/videos/<task>/repeat_<test_count>/          # MP4s, when trainer.store_videos
```

(`rollout_artifact_dir` builds both; `test_count` is the `times_repeat` round index). Nesting keeps a 24-task × 200-scene eval browsable. Each PNG has two panels:

- **Left — aggregate presence graph**: networkx spring layout, nodes colored viridis by presence %, sized proportionally, with `name\n%` labels. Nodes represent the *union* of all selections over the rollout and may therefore exceed K.
- **Right — per-step selection heatmap**: binary matrix (rows = objects sorted by presence, columns = timesteps). Each column sums to ≤ K, directly showing which K nodes were active at every step and how the selection set evolves over time.

`Multi_GNN` has no `last_sparsification` attribute; the accumulation block is guarded by `hasattr`, so no `sparsification/` folder is created for the baseline encoder.

## Rollout Sub-Graph Faithfulness

The same accumulation block scores the kept sub-graph against the task's ground-truth relevant nodes, answering *how faithful is the selection*, not just *how sparse*. Per inference step, `score_subgraph_faithfulness` compares the selected node names to the relevant ones and returns **precision / recall / F1 / Jaccard**; the recall denominator is the relevant nodes *present in the graph*, since the model cannot select what it never observed. Unlike the training-time CBM loss — which receives no `lang_goal_texts` at rollout — this runs where the raw language goal is available, so the language-resolved door node of the `Open/Close*Door` tasks is scored correctly.

Under VGIB `eval_prune_mode: soft`, note that the reported selection is the threshold cut while the policy actually acted on *every* (attenuated) node — the faithfulness number there describes the ranking, not the graph the policy consumed.

Scores are averaged over a rollout's steps, so every rollout weighs equally regardless of length. The manager then aggregates the `eval_n_times × times_repeat` grid two ways (`_summarize_faithfulness` in [manager/robocasa_manager.py](manager/robocasa_manager.py)):

- **per repeat-round** — mean over that round's scenes (`times_repeat` values)
- **per scene** — mean over the rounds that ran that scene (`eval_n_times` values)

wandb keys (logged by [trainer.py](trainer.py) alongside `Avg_Test_Reward`, with the usual `test_during_training_` prefix during in-training eval):

- `faithfulness/{mod}/{task}/round_{i}_{metric}` — per repeat-round
- `faithfulness/{mod}/{task}/{metric}` — over all of the task's rollouts
- `faithfulness/{mod}/{metric}` — unweighted mean over tasks
- `faithfulness_per_scene` — a `wandb.Table` carrying the per-scene grouping (one row per task × modality × scene), which would otherwise be thousands of unusable scalar series

The full round × scene grid is also written to `<working_dir>/eval_faithfulness_<task>[_epoch<e>].json`, beside `eval_per_scene_<task>.json`. These values travel on `manager.last_faithfulness_log` rather than in the `test_method` result dict — every key in that dict is summed into `Avg_Test_Reward`, so an extra key there would corrupt the headline success number.

## Parallel Evaluation

Rollouts, not training, dominate wall-clock on a 24-task sweep, so `test_method` has an opt-in policy-server path ([envs/robocasa/parallel/](envs/robocasa/parallel/)). When `manager.parallel_eval` is true and `num_envs > 1`, it spawns `num_envs` MuJoCo env-worker processes (one CPU core each) and a GPU-resident policy-server thread in the main process; observations are batched per forward pass. Single-machine only; with `num_envs == 1` or `parallel_eval: false` it falls back to the sequential `RoboCasaKitchenTester`, and both paths expose the same `per_scene_success` / `per_scene_faithfulness` / `node_scores`.

Knobs: `num_envs`, `batch_timeout_ms` (how long the server waits to accumulate a batch), `max_eval_batch` (hard per-forward cap), `worker_seed_base` (per-worker seed = `base + env_id*100003 + hash(task)`; held constant across rounds so rollout *j* of a task sees the same scene every round, matching the sequential tester).

**Sizing.** Each worker's EGL context takes ~2.0 GB of VRAM and varies ~25% between workers. The committed default of `num_envs: 2` is sized for an 8 GB card, where 3 workers peaked at 93% and the policy server OOMed mid-evaluation. Raise it on a bigger card.

**Degraded-evaluation detection.** A failed server forward is answered with zero actions so the workers do not hang. That keeps the run alive but silently degrades whatever rollouts were in flight — nothing crashes, the rollout count does not shorten, and the run just comes back quietly worse (a single CUDA OOM did exactly this on 2026-08-26). The server therefore counts `zero_action_replies` and records the first error, and `RoboCasaKitchenParallelTester` emits a loud `DEGRADED EVALUATION` log line stating that every `Avg_Test_Reward` in that round is a lower bound.

## Run naming

Every run is "some variant at some seed", but that identity is spread across four config keys (the graph encoder's target class, its sampling strategy, whether the CBM loss is on, and which modality lists are non-empty). [utils/run_naming.py](utils/run_naming.py) computes it from the config that actually ran, so the label cannot disagree with the run:

`<Variant>[_<dataset>]_s<seed>` → `SIR_s42`, `VGIB_s43`, `TopK_s44`, `Threshold_s42`, `GNN_s42`, `Image_s42`, `Prop_s42`, `VGIB+Image_s42`.

Two deliberate distinctions: `topk_PerTask` **with** the CBM loss is `SIR`, without it is `PerTask` (the budget alone is only half the method); `vgib` **with** a CBM loss is `VGIB+CBM`, because a run with both is neither the unsupervised variant nor SIR. An explicit `wandb.run_name=` still wins. `wandb.mode` (`online` / `offline` / `disabled`) is config-driven so a run needs no wandb login and no source edit.

## Training & Evaluation Flow

1. **Data prep**: the manager calls `check_and_create_graph_dataset` to build the requested per-modality `.pth` graph tensors under `data_path/<task>/` if missing (`point_geom` excepted — it raises with the replay command instead).
2. **Dataset**: [dataloader/datasets.py](dataloader/datasets.py) loads HDF5 demos (`demo_gentex_im128_randcams.hdf5`), per-demo 128×128 image tensors (stored **uint8**: inverting the export normalisation lands them exactly on the 8-bit grid, verified bit-identical on all 24 tasks, and float32 would need 211 GB resident for the 24-task image baseline against 123 GB of RAM — note the export used blue-channel std **0.255**, unlike the 0.225 the graph-crop denormalisation uses), and the precomputed graphs, then serves obs-window / action-window slices with CLIP-encoded language goals. Raw language goal text is also stored in `item['goal']['lang_text']` for the CBM loss. `manager.num_train_demos` caps how many demos per task reach the **train** split (null = all, ~51 after `train_per`); the validation split stays the full `1 - train_per` tail so a low-data run and a full-data run are compared on the same held-out demos, and demos are taken in sorted order so the subset is deterministic.
3. **Training**: [trainer.py](trainer.py) runs validation-then-training per epoch, with optional EMA, a warmup + step LR schedule, optional gradient clipping, per-epoch node-score snapshots, and periodic in-sim rollouts.
4. **Testing**: the sequential or parallel tester resets each kitchen environment, reconstructs the same observation modalities on-the-fly from live ground-truth segmentation masks, rolls out the diffusion policy, and records success rates + sub-graph faithfulness + node scores + MP4 videos + sparsification PNGs (the last two nested per task and repeat round).

## Configuration Entry Points

- [configs/main.yaml](configs/main.yaml) — trainer / wandb / seeds / log paths, `graph_eval_mode`, `task_thresholds`, `init_from*`
- [configs/manager/robocasa.yaml](configs/manager/robocasa.yaml) — tasks, batch size, `num_train_demos`, modality selection (`graph_modalities`, `use_graph_fusion`, `use_splitted_modalities`), parallel-eval block
- [configs/method/diffusion.yaml](configs/method/diffusion.yaml) — diffusion hyperparameters, optimizer, LR scheduler
- [configs/method/action_generator/mdt.yaml](configs/method/action_generator/mdt.yaml) — transformer depth/width, `max_graph_nodes`
- [configs/method/graph_encoder/gnn.yaml](configs/method/graph_encoder/gnn.yaml) — standard GATv2 encoder (`Multi_GNN`)
- [configs/method/graph_encoder/xai_gnn.yaml](configs/method/graph_encoder/xai_gnn.yaml) — per-task top-k, CBM on → **SIR**
- [configs/method/graph_encoder/xai_gnn_topk.yaml](configs/method/graph_encoder/xai_gnn_topk.yaml) — task-agnostic fixed-k (default k=5, CBM off)
- [configs/method/graph_encoder/xai_gnn_threshold.yaml](configs/method/graph_encoder/xai_gnn_threshold.yaml) — threshold-based selection (default 0.5, CBM off)
- [configs/method/graph_encoder/xai_gnn_vgib.yaml](configs/method/graph_encoder/xai_gnn_vgib.yaml) — VGIB stage 1 (unsupervised)
- [configs/method/graph_encoder/xai_gnn_vgib_stage2.yaml](configs/method/graph_encoder/xai_gnn_vgib_stage2.yaml) — VGIB stage 2 (whitelist distillation)
- [configs/method/vision_encoder/cnn.yaml](configs/method/vision_encoder/cnn.yaml) — image encoder

## Supported Tasks

24 RoboCasa kitchen tasks grouped into Drawer, Doors, Microwave, Sink, Stove, Coffee, and Pick-and-Place (PnP) families — see `TASK_LIST` in [envs/robocasa/utils.py](envs/robocasa/utils.py). Training can target a single task, a subset, or `"ALL"`.

## Tests

Plain-python test files under [tests/](tests/) (no pytest required — `python tests/test_vgib.py`): `test_vgib` (VGIB KL modes, whitelist, per-task thresholds, and the assertion that the VGIB path never touches `RELEVENT_NODES`), `test_node_scores`, `test_run_naming`, `test_gnn_path`, `test_mdt_path`, `test_graph_fusion`, `test_prop_obs`, `test_point_geom`.

## Setup

See [README.md](README.md) and [install.sh](install.sh). Installation requires patched forks of `robosuite` (branch `robocasa_v0.1`) and `robocasa` (pinned commit) in sibling directories, with the exact source-code edits listed in the README (segmentation support, deterministic seeding, and contact-geom cleanup). SLURM training launcher is at [slurm_scripts/train.sh](slurm_scripts/train.sh).
