"""Tests for the VGIB sparsifier (Yu et al., CVPR 2022).

VGIB exists in this repo to answer one question: does a sparsifier that has never seen
`RELEVENT_NODES` pick the same nodes the CBM-supervised one does? That framing makes two
properties load-bearing, and both are easy to break silently:

  1. The noised features must actually reach the GNN. If `x_out` is dropped on the floor
     the compression KL is computed against a corruption the model never sees, the run
     trains, the loss curve looks fine, and the whole thing measures nothing.
  2. Nothing on the VGIB path may consult the relevance table. A single call to
     `get_num_relevant_nodes_per_task` — which the legacy `uniform_except_middle` loss
     makes without looking like it does — turns the unsupervised result back into the
     supervised one.

Run: python tests/test_vgib.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Batch, Data

import networks.graph_encoder.gnn as gnn_module
from networks.graph_encoder.gnn import Multi_XAI_GNN, Sparsification_Module

IN_DIM = 8
HIDDEN = 16
LANG_DIM = HIDDEN


def _batch(sizes=(4, 6)):
    datas = []
    for N in sizes:
        x = torch.randn(N, IN_DIM)
        src, dst = zip(*[(i, j) for i in range(N) for j in range(N) if i != j])
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.int64)
        datas.append(Data(x=x, edge_index=edge_index,
                          edge_attr=torch.ones(edge_index.shape[1])))
    return Batch.from_data_list(datas)


def _spars(**kw):
    cfg = dict(in_dim=IN_DIM, hidden_dim=HIDDEN, num_heads=2, num_layers=1,
               dropout_prob=0.0, sampling_strategy="vgib", threshold=0.5,
               vgib_noise_kl_weight=1e-2, vgib_noise_kl_mode="full",
               eval_prune_mode="hard")
    cfg.update(kw)
    return Sparsification_Module(**cfg)


def _run(spars, batch, train=True, lang=None):
    spars.train(train)
    if lang is None:
        lang = torch.randn(batch.num_graphs, LANG_DIM)
    return spars(batch.x, batch.edge_index, edge_attr=None, edge_weights=batch.edge_attr,
                 lang_emb=lang, batch=batch, task_names=["CloseDrawer"] * batch.num_graphs)


# --------------------------------------------------------------- the unsupervised claim

def test_vgib_never_consults_the_relevance_table():
    """The property the variant is for. `uniform_except_middle` sizes itself from
    `get_num_relevant_nodes_per_task`, so leaving the legacy loss loop running would put
    the hand-written table back into a path that claims not to use it."""
    calls = []
    original = gnn_module.get_num_relevant_nodes_per_task
    gnn_module.get_num_relevant_nodes_per_task = lambda t: (calls.append(t), original(t))[1]
    try:
        spars = _spars(uniform_except_middle_weight=1.0, uniform_dist_weight=1.0)
        _run(spars, _batch(), train=True)
    finally:
        gnn_module.get_num_relevant_nodes_per_task = original
    assert calls == [], f"VGIB path consulted RELEVENT_NODES for {calls}"


def test_topk_per_task_does_consult_it():
    """Control for the test above — if the table were unreachable in general, that test
    would pass for the wrong reason."""
    calls = []
    original = gnn_module.get_num_relevant_nodes_per_task
    gnn_module.get_num_relevant_nodes_per_task = lambda t: (calls.append(t), original(t))[1]
    try:
        spars = _spars(sampling_strategy="topk_PerTask", vgib_noise_kl_weight=0.0)
        _run(spars, _batch(), train=True)
    finally:
        gnn_module.get_num_relevant_nodes_per_task = original
    assert calls, "topk_PerTask no longer uses the relevance table — the control is broken"


# ------------------------------------------------------------------ information filtration

def test_training_returns_noised_features_and_prunes_nothing():
    batch = _batch()
    edge_w, sampled, probs, x_out, raw = _run(_spars(), batch, train=True)
    assert x_out is not None and x_out.shape == batch.x.shape
    assert not torch.allclose(x_out, batch.x), "features were not noised"
    # Discarding happens by dissolving into noise, not by cutting edges.
    assert torch.allclose(edge_w, batch.edge_attr), "soft path gated the topology"
    assert (probs > 0).all(), "soft path hard-zeroed scores"


def test_soft_eval_does_not_threshold_anything():
    """The mode a sweep is read off. Nothing is cut, edges are untouched, and every node
    is attenuated toward the graph mean exactly as in training — so the measured number
    reflects the learned scores rather than the scores plus a threshold chosen after."""
    batch = _batch()
    spars = _spars(eval_prune_mode="soft")
    lang = torch.randn(batch.num_graphs, LANG_DIM)
    edge_w, _, probs, x_out, raw = _run(spars, batch, train=False, lang=lang)
    assert x_out is not None, "soft eval must still attenuate the features"
    assert torch.allclose(edge_w, batch.edge_attr), "soft eval gated the topology"
    assert torch.allclose(probs, raw), "soft eval thresholded the scores"
    # Deterministic: the expectation, not a fresh draw.
    _, _, _, x_out2, _ = _run(spars, batch, train=False, lang=lang)
    assert torch.allclose(x_out, x_out2), "soft eval is stochastic"


def test_noise_eval_replaces_dropped_nodes_but_keeps_the_graph():
    batch = _batch()
    spars = _spars(eval_prune_mode="noise", threshold=0.5)
    edge_w, _, probs, x_out, raw = _run(spars, batch, train=False)
    assert x_out is not None and x_out.shape == batch.x.shape
    assert torch.allclose(edge_w, batch.edge_attr), "noise eval cut edges"
    assert torch.allclose(probs, raw), "noise eval zeroed scores instead of noising"
    kept = raw > spars.threshold
    if bool(kept.any()):
        assert torch.allclose(x_out[kept], batch.x[kept]), "kept nodes were altered"
    if bool((~kept).any()):
        assert not torch.allclose(x_out[~kept], batch.x[~kept]), "dropped nodes untouched"


def test_the_three_eval_modes_disagree():
    """If they did not, the mode would not be worth having — and the 0.600/0.185 split
    the first sweep produced would have been unexplainable."""
    batch = _batch()
    outs = {}
    lang = torch.randn(batch.num_graphs, LANG_DIM)
    for mode in ("soft", "noise", "hard"):
        spars = _spars(eval_prune_mode=mode)
        torch.manual_seed(0)
        _, sampled, probs, x_out, _ = _run(spars, batch, train=False, lang=lang)
        outs[mode] = (probs.clone(), None if x_out is None else x_out.clone())
    assert outs["hard"][1] is None and outs["soft"][1] is not None
    assert not torch.allclose(outs["soft"][0], outs["hard"][0]), "soft == hard scores"


def test_eval_is_deterministic_and_prunes():
    batch = _batch()
    spars = _spars(eval_prune_mode="hard")
    # Same language embedding both times — the scorer is conditioned on it, so a fresh
    # draw would make this test measure nothing but its own randomness.
    lang = torch.randn(batch.num_graphs, LANG_DIM)
    _, s1, p1, x1, _ = _run(spars, batch, train=False, lang=lang)
    _, s2, p2, x2, _ = _run(spars, batch, train=False, lang=lang)
    assert x1 is None and x2 is None, "eval must not noise"
    assert torch.equal(s1, s2) and torch.allclose(p1, p2), "eval is not deterministic"
    assert bool(((p1 == 0) | (p1 > spars.threshold)).all()), "eval left sub-threshold scores alive"


def test_noise_interpolates_toward_the_graph_mean():
    """p=1 keeps a node exactly; p=0 replaces it with a draw from the graph's own
    Gaussian. Checked at the deterministic p=1 end."""
    batch = _batch(sizes=(5,))
    spars = _spars()
    spars.train(True)
    with torch.no_grad():
        for m in spars.coarsening_layer.scorer.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.zero_()
                m.bias.fill_(20.0)  # sigmoid -> 1.0
    _, _, probs, x_out, _ = _run(spars, batch, train=True)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)
    assert torch.allclose(x_out, batch.x, atol=1e-3), "p=1 did not keep the features"


# ---------------------------------------------------------------------- the KL itself

def _kl(mode, p_value, n=6):
    spars = _spars(vgib_noise_kl_mode=mode)
    x = torch.randn(n, IN_DIM)
    probs = torch.full((n,), p_value)
    node_mask = torch.zeros(n, dtype=torch.long)
    spars.coarsening_loss = {}
    spars.train(True)
    spars._vgib(x, probs, node_mask, 1)
    return float(spars.coarsening_loss['vgib_noise_kl_loss']['value'])


def test_full_kl_is_monotone_in_p():
    """The reason `full` is the default: it is pure down-pressure, so only the prediction
    loss can hold a node up. `truncated` drops the -log(1-p) term, which leaves an
    interior fixed point where nodes park mid-scale instead of committing."""
    values = [_kl("full", p) for p in (0.01, 0.25, 0.5, 0.75, 0.95)]
    assert all(b > a for a, b in zip(values, values[1:])), f"not monotone: {values}"


def test_full_kl_vanishes_at_p_zero():
    assert abs(_kl("full", 1e-6)) < 1e-3


def test_every_mode_produces_a_finite_loss():
    for mode in ("full", "truncated", "eq10"):
        v = _kl(mode, 0.6)
        assert v == v and abs(v) != float('inf'), f"{mode} produced {v}"


def test_rent_only_charges_above_the_threshold():
    spars = _spars(threshold=0.5, threshold_sum_weight=1.0)
    x = torch.randn(4, IN_DIM)
    node_mask = torch.zeros(4, dtype=torch.long)
    spars.train(True)

    spars.coarsening_loss = {}
    spars._vgib(x, torch.tensor([0.1, 0.2, 0.3, 0.4]), node_mask, 1)
    assert float(spars.coarsening_loss['coars_threshold_sum_loss']['value']) == 0.0

    spars.coarsening_loss = {}
    spars._vgib(x, torch.tensor([0.9, 0.2, 0.3, 0.4]), node_mask, 1)
    assert abs(float(spars.coarsening_loss['coars_threshold_sum_loss']['value']) - 0.9) < 1e-6


# --------------------------------------------------------------- end-to-end wiring

def _encoder(**kw):
    spars_cfg = {
        '_target_': 'networks.graph_encoder.gnn.Sparsification_Module',
        '_recursive_': False,
        'in_dim': {}, 'hidden_dim': HIDDEN, 'num_heads': 2, 'num_layers': 1,
        'dropout_prob': 0.0, 'sampling_strategy': 'vgib', 'threshold': 0.5,
        'vgib_noise_kl_weight': 1e-2, 'vgib_noise_kl_mode': 'full',
    }
    spars_cfg.update(kw)
    from omegaconf import OmegaConf
    return Multi_XAI_GNN(
        sparsification_layer=OmegaConf.create(spars_cfg),
        input_dim={'bb_coordinates': IN_DIM}, hidden_dim=HIDDEN, output_dim=HIDDEN,
        edge_dim=None, num_layer=1, layer_name='WeightedGATv2', pool_name='mean',
        heads=2, dropout=0.0, modalities=['bb_coordinates'],
        use_cbm_loss=None, lcbm_weight=0.0, multiplicative_pooling=True,
    )


def test_noised_features_reach_the_gnn():
    """Bug 1 from the module docstring: if `x_out` is not written back to `batch.x`, the
    KL is paid on a corruption the GNN never sees and the run trains nothing."""
    enc = _encoder()
    enc.train(True)
    batch = _batch()
    original = batch.x.clone()
    enc({'bb_coordinates': batch}, lang_emb=torch.randn(batch.num_graphs, LANG_DIM),
        task_names=['CloseDrawer'] * batch.num_graphs)
    assert not torch.allclose(batch.x, original), "batch.x was never replaced by x_out"


def test_kl_is_registered_as_an_aux_loss_with_its_weight():
    enc = _encoder()
    enc.train(True)
    batch = _batch()
    enc({'bb_coordinates': batch}, lang_emb=torch.randn(batch.num_graphs, LANG_DIM),
        task_names=['CloseDrawer'] * batch.num_graphs)
    key = 'bb_coordinates/vgib_noise_kl_loss'
    assert key in enc.aux_losses, list(enc.aux_losses)
    assert enc.aux_losses[key]['weight'] == 1e-2
    assert enc.aux_losses[key]['value'].requires_grad, "KL is detached from the scorer"


def test_gradient_reaches_the_scorer():
    enc = _encoder()
    enc.train(True)
    batch = _batch()
    out = enc({'bb_coordinates': batch}, lang_emb=torch.randn(batch.num_graphs, LANG_DIM),
              task_names=['CloseDrawer'] * batch.num_graphs)
    emb = out['bb_coordinates']
    emb = emb[0] if isinstance(emb, tuple) else emb
    loss = emb.sum() + enc.aux_losses['bb_coordinates/vgib_noise_kl_loss']['value']
    loss.backward()
    scorer = enc.sparsification_modules['bb_coordinates'].coarsening_layer.scorer
    grads = [p.grad for p in scorer.parameters() if p.grad is not None]
    assert grads and any(bool((g.abs() > 0).any()) for g in grads), "scorer got no gradient"


def test_raw_probs_survive_the_selection_mask():
    """The threshold has to be chosen by ranking the *rejected* nodes too, so the
    pre-mask scores must be reported alongside the masked ones."""
    enc = _encoder()
    enc.train(False)
    batch = _batch()
    enc({'bb_coordinates': batch}, lang_emb=torch.randn(batch.num_graphs, LANG_DIM),
        task_names=['CloseDrawer'] * batch.num_graphs)
    info = enc.last_sparsification['bb_coordinates']
    assert bool((info['raw_probs'] > 0).all()), "raw scores were masked"
    assert info['task_names'] == ['CloseDrawer'] * batch.num_graphs


# ------------------------------------------------- the property that picks the KL mode

def test_truncated_has_an_interior_optimum_and_full_does_not():
    """The reason the branch uses `truncated`. Its per-node optimum sits at
    p* = 1/(1+d^2) — a readout of how far the node deviates from the graph mean, so the
    scores stay graded and rankable. `full` optimises every node toward 0 regardless of
    d, so nothing survives except what the prediction loss holds up; and that loss barely
    moves when sparsification destroys a rollout, which is how a whole sweep ends up with
    no gap to cut at."""
    import torch as T
    for d in (0.5, 1.0, 2.0):
        ps = T.linspace(0.001, 0.999, 999)
        quad = 0.5 * ((1 - ps) ** 2 + (ps * d) ** 2)
        trunc_star = float(ps[int(quad.argmin())])
        full = -T.log1p(-ps.clamp(max=1 - 1e-6)) + quad - 0.5
        full_star = float(ps[int(full.argmin())])
        assert abs(trunc_star - 1 / (1 + d * d)) < 0.01, f"d={d}: p*={trunc_star}"
        assert full_star < 0.01, f"d={d}: full mode kept an interior optimum at {full_star}"


# --------------------------------------------------------------------- stage 2

def _wl_spars(**kw):
    return _spars(sampling_strategy="threshold", threshold=0.5,
                  vgib_noise_kl_weight=0.0,
                  node_whitelist={"CloseDrawer": ["n0", "n1"]},
                  whitelist_bce_weight=0.1, whitelist_pos_weight=4.3, **kw)


def _named_batch(sizes=(4, 4)):
    batch = _batch(sizes)
    batch.node_names = [[f"n{i}" for i in range(n)] for n in sizes]
    return batch


def test_whitelist_bce_is_registered_and_supervises_the_scorer():
    spars = _wl_spars()
    batch = _named_batch()
    _run(spars, batch, train=True)
    entry = spars.coarsening_loss.get("whitelist_bce_loss")
    assert entry is not None, list(spars.coarsening_loss)
    assert entry["weight"] == 0.1
    assert entry["value"].requires_grad


def test_whitelist_match_is_reported():
    spars = _wl_spars()
    _run(spars, _named_batch(), train=True)
    m = spars._last_whitelist_match
    assert m is not None and 0.0 <= m <= 1.0


def test_stage1_reports_no_whitelist_match():
    """The canary must be absent, not 0.0, when there is no whitelist — a 0.0 would read
    as 'the scorer disagrees with everything'."""
    spars = _spars()
    _run(spars, _batch(), train=True)
    assert spars._last_whitelist_match is None


def test_fixed_strategy_uses_the_whitelist_as_the_mask():
    spars = _spars(sampling_strategy="fixed", vgib_noise_kl_weight=0.0,
                   node_whitelist={"CloseDrawer": ["n0", "n2"]})
    batch = _named_batch(sizes=(4,))
    _, sampled, probs, x_out, _ = _run(spars, batch, train=True)
    assert x_out is None, "fixed must not noise — the subgraph is already decided"
    assert sampled.tolist() == [0, 2]
    assert probs.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_fixed_without_a_whitelist_is_rejected():
    try:
        _spars(sampling_strategy="fixed")
    except ValueError:
        return
    raise AssertionError("sampling_strategy 'fixed' was accepted with no whitelist")


def test_bce_on_top_of_fixed_is_rejected():
    """Under `fixed` the whitelist is already the mask, so supervising toward it trains
    nothing — silently, which is worse than failing."""
    try:
        _spars(sampling_strategy="fixed", node_whitelist={"CloseDrawer": ["n0"]},
               whitelist_bce_weight=0.1)
    except ValueError:
        return
    raise AssertionError("whitelist BCE was accepted under 'fixed'")


def test_bce_without_a_whitelist_is_rejected():
    try:
        _spars(sampling_strategy="threshold", whitelist_bce_weight=0.1)
    except ValueError:
        return
    raise AssertionError("whitelist BCE was accepted with no whitelist")


def test_unknown_task_in_whitelist_fails_loudly():
    """A task missing from the whitelist would otherwise train against an all-zero
    target, i.e. teach the policy to go scene-blind on it."""
    spars = _wl_spars()
    batch = _named_batch(sizes=(4,))
    try:
        spars.train(True)
        spars(batch.x, batch.edge_index, edge_attr=None, edge_weights=batch.edge_attr,
              lang_emb=torch.randn(1, LANG_DIM), batch=batch, task_names=["NotATask"])
    except KeyError:
        return
    raise AssertionError("an unknown task silently produced an empty subgraph")


# ------------------------------------------------------ eval-only ablations

def test_graph_swap_full_keeps_every_node():
    spars = _spars(eval_prune_mode="hard")
    spars.graph_swap = "full"
    batch = _batch()
    _, sampled, probs, _, _ = _run(spars, batch, train=False)
    assert sampled.numel() == batch.x.shape[0], "full did not keep every node"
    assert bool((probs == 1).all())


def test_graph_swap_random_matches_the_learned_size():
    """The control that makes a subgraph result mean anything: same size, different
    nodes. If a random subgraph scores as well, the identity of the nodes was never
    what mattered."""
    batch = _batch(sizes=(8, 8))
    lang = torch.randn(batch.num_graphs, LANG_DIM)
    spars = _spars(eval_prune_mode="hard", threshold=0.5)
    _, learned, _, _, _ = _run(spars, batch, train=False, lang=lang)
    spars.graph_swap = "random"
    _, swapped, probs, _, _ = _run(spars, batch, train=False, lang=lang)
    assert swapped.numel() == learned.numel(), \
        f"random kept {swapped.numel()} nodes against the learned {learned.numel()}"
    assert set(probs.tolist()) <= {0.0, 1.0}, "random selection must be hard 0/1"


def test_random_subgraph_is_fixed_per_scene_not_per_timestep():
    """The learned subgraph is stable for as long as the scene is. A random control
    redrawn every step would differ from it in two ways at once — wrong nodes AND
    flicker — so a drop could not be attributed to either."""
    batch = _batch(sizes=(8,))
    batch.node_names = [[f"obj{i}" for i in range(8)]]
    spars = _spars(eval_prune_mode="hard", threshold=0.5)
    spars.graph_swap = "random"
    lang = torch.randn(1, LANG_DIM)
    picks = set()
    for _ in range(5):   # five "timesteps" on the same scene
        _, sampled, _, _, _ = _run(spars, batch, train=False, lang=lang)
        picks.add(tuple(sampled.tolist()))
    assert len(picks) == 1, f"the random subgraph changed between steps: {picks}"


def test_random_subgraph_differs_between_scenes():
    """Fixed per scene, but not the same subgraph everywhere — otherwise it is one
    arbitrary choice rather than a control."""
    spars = _spars(eval_prune_mode="hard", threshold=0.5)
    spars.graph_swap = "random"
    lang = torch.randn(1, LANG_DIM)
    picks = set()
    for scene in range(6):
        batch = _batch(sizes=(8,))
        batch.node_names = [[f"s{scene}_obj{i}" for i in range(8)]]
        _, sampled, _, _, _ = _run(spars, batch, train=False, lang=lang)
        picks.add(tuple(sampled.tolist()))
    assert len(picks) > 1, "every scene drew the same random subgraph"


def test_graph_swap_seed_changes_the_draw():
    batch = _batch(sizes=(8,))
    batch.node_names = [[f"obj{i}" for i in range(8)]]
    lang = torch.randn(1, LANG_DIM)
    out = []
    for seed in (0, 1):
        spars = _spars(eval_prune_mode="hard", threshold=0.5)
        spars.graph_swap, spars.graph_swap_seed = "random", seed
        _, sampled, _, _, _ = _run(spars, batch, train=False, lang=lang)
        out.append(tuple(sampled.tolist()))
    assert out[0] != out[1], "graph_swap_seed did not change the subgraph"


def test_graph_swap_is_eval_only():
    spars = _spars()
    spars.graph_swap = "full"
    _, sampled, _, x_out, _ = _run(spars, _batch(), train=True)
    assert x_out is not None, "the swap leaked into training"


def test_unknown_graph_swap_is_rejected():
    enc = _encoder()
    try:
        enc.set_graph_swap("nonsense")
    except AssertionError:
        return
    raise AssertionError("an unknown graph_swap mode was accepted")


# ------------------------------------------------------ per-task thresholds

def test_task_thresholds_cut_each_task_at_its_own_bar():
    """The measured gap sits at 0.878 on CloseDrawer and 0.797 on CoffeePressButton, so
    a single global bar necessarily over-prunes one and under-prunes the other."""
    batch = _batch(sizes=(6, 6))
    lang = torch.randn(2, LANG_DIM)
    spars = _spars(eval_prune_mode="hard", threshold=0.5)
    spars.train(False)
    _, _, _, _, raw = spars(batch.x, batch.edge_index, edge_attr=None,
                            edge_weights=batch.edge_attr, lang_emb=lang, batch=batch,
                            task_names=["CloseDrawer", "CoffeePressButton"])
    lo, hi = 0.0, 1.0
    spars.task_thresholds = {"CloseDrawer": hi, "CoffeePressButton": lo}
    _, sampled, _, _, _ = spars(batch.x, batch.edge_index, edge_attr=None,
                                edge_weights=batch.edge_attr, lang_emb=lang, batch=batch,
                                task_names=["CloseDrawer", "CoffeePressButton"])
    # Graph 0 is cut at 1.0 (nothing survives), graph 1 at 0.0 (everything does).
    assert all(i >= 6 for i in sampled.tolist()), \
        "the per-task bar was not applied per graph"
    assert sampled.numel() == 6


def test_task_thresholds_do_not_apply_during_training():
    """Training must keep the scalar bar the losses were calibrated against."""
    spars = _spars(eval_prune_mode="hard", threshold=0.5)
    spars.task_thresholds = {"CloseDrawer": 1.0}
    _, sampled, _, _, _ = _run(spars, _batch(sizes=(6,)), train=True)
    assert sampled.numel() > 0, "an eval-only threshold leaked into training"


def test_unlisted_tasks_fall_back_to_the_scalar():
    spars = _spars(eval_prune_mode="hard", threshold=0.0)
    spars.task_thresholds = {"SomeOtherTask": 1.0}
    _, sampled, _, _, _ = _run(spars, _batch(sizes=(5,)), train=False)
    assert sampled.numel() == 5


# ------------------------------------------------------------------------- config

def test_stage2_config_is_still_unsupervised_by_the_relevance_table():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/method/graph_encoder/xai_gnn_vgib_stage2.yaml')
    assert cfg.use_cbm_loss is None and cfg.lcbm_weight == 0.0
    assert cfg.sparsification_layer.sampling_strategy == 'threshold'
    assert cfg.sparsification_layer.whitelist_bce_weight == 0.1
    assert cfg.sparsification_layer.vgib_noise_kl_weight == 0.0


def test_shipped_config_is_unsupervised_and_instantiable():
    import hydra
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/method/graph_encoder/xai_gnn_vgib.yaml')
    assert cfg.use_cbm_loss is None, "the shipped VGIB config turned the CBM loss back on"
    assert cfg.lcbm_weight == 0.0
    assert cfg.sparsification_layer.sampling_strategy == 'vgib'
    # Every heuristic the KL replaces must be off, or the comparison is confounded.
    for k in ('variance_weight', 'sum_mask_weight', 'entropy_reg_weight',
              'uniform_dist_weight', 'uniform_except_middle_weight'):
        assert cfg.sparsification_layer[k] == 0.0, f"{k} is not disabled"

    spars_cfg = OmegaConf.to_container(cfg.sparsification_layer, resolve=True)
    spars_cfg['in_dim'] = IN_DIM
    module = hydra.utils.instantiate(OmegaConf.create(spars_cfg))
    assert isinstance(module, Sparsification_Module)
    # `truncated`, not `full`: only the truncated form has an interior optimum, which is
    # what leaves the scores graded enough to threshold. See the config header.
    assert module.vgib_noise_kl_mode == 'truncated'
    assert module.eval_prune_mode == 'soft'
    assert module.node_whitelist is None, "stage 1 must not carry a whitelist"
    assert module.whitelist_bce_weight == 0.0


def test_unknown_eval_prune_mode_is_rejected():
    try:
        _spars(eval_prune_mode='nonsense')
    except AssertionError:
        return
    raise AssertionError("an unknown eval_prune_mode was accepted")


def test_unknown_kl_mode_is_rejected():
    try:
        _spars(vgib_noise_kl_mode='nonsense')
    except AssertionError:
        return
    raise AssertionError("an unknown vgib_noise_kl_mode was accepted")
# --- uniform_dist_weight must be reachable from the vgib path ---------------------------
#
# The histogram loss was previously skipped wholesale under `vgib`, so setting
# `uniform_dist_weight` on a VGIB run was a silent no-op: the run trained, logged the key
# at 0.0, and looked healthy. The distinction that matters is per-term, not per-strategy —
# only `uniform_except_middle` reads the relevance table, and it is the only one that must
# stay off here. These drive the real forward() and read what it registered.

def test_uniform_dist_weight_survives_vgib():
    """The histogram loss is score-only, so vgib must be allowed to use it."""
    spars = _spars(uniform_dist_weight=1e-1)
    _run(spars, _batch(), train=True)
    term = spars.coarsening_loss['coars_uniform_loss']
    assert term['weight'] == 1e-1, \
        f"uniform_dist_weight was zeroed under vgib (got {term['weight']}) -- silent no-op"
    assert float(term['value']) != 0.0, \
        "the histogram loss registered at exactly 0.0 -- it was never computed"


def test_uniform_dist_weight_reaches_the_gradient():
    """Registered-but-detached would be just as silent a failure as not computing it."""
    spars = _spars(uniform_dist_weight=1e-1)
    _run(spars, _batch(), train=True)
    value = spars.coarsening_loss['coars_uniform_loss']['value']
    assert value.requires_grad, "histogram loss carries no grad_fn -- scorer cannot learn from it"


def test_uniform_except_middle_forced_off_under_vgib():
    """This one sizes itself from the relevance table, so vgib must never compute it --
    even when the config explicitly asks for it."""
    spars = _spars(uniform_except_middle_weight=1e-1)
    _run(spars, _batch(), train=True)
    term = spars.coarsening_loss['coars_uniform_except_middle_loss']
    assert term['weight'] == 0.0 and float(term['value']) == 0.0, \
        "the relevance table leaked into the unsupervised path"


def test_except_middle_still_active_for_supervised_strategies():
    """Control: forcing it off must be specific to vgib, not global."""
    spars = _spars(sampling_strategy="topk_PerTask", vgib_noise_kl_weight=0.0,
                   uniform_except_middle_weight=1e-1)
    _run(spars, _batch(), train=True)
    assert spars.coarsening_loss['coars_uniform_except_middle_loss']['weight'] == 1e-1


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [name for name in dir() if name.startswith("test_")]
    failed = []
    for name in tests:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    sys.exit(0 if not failed else 1)
