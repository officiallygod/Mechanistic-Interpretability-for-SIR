"""Masked-node leakage audit for a subgraph-selection explainability method.

THE AUDIT, stated generally
---------------------------
A masking-based method claims: *the prediction depends only on the selected
subgraph*. That claim is testable without any knowledge of the architecture:

    1. run the model on a graph, with a mask marking kept/pruned nodes
    2. replace the FEATURES of the pruned nodes with arbitrary different values
    3. re-run with the identical mask
    4. every quantity the downstream policy consumes must be BIT-IDENTICAL

Any non-zero difference is a channel by which a node the method says it removed
still reaches the prediction. The perturbation is deliberately huge (randn * 50):
a leak scales with it, round-off does not.

This file both *measures* the leak (so the size can be reported) and *asserts*
the guarantee holds once the closing flags are set.

Run: python tests/test_masking_leak.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data, Batch

from networks.graph_encoder.gnn import GNN

_MOD = "cropped_image_feature_bb_coordinates"

INPUT_DIM, HIDDEN, OUTPUT = 12, 32, 16
N_NODES, N_GRAPHS = 6, 2
KEEP = 3                      # first KEEP nodes of each graph survive


def _batch(seed=0, num_graphs=N_GRAPHS, n=N_NODES):
    g = torch.Generator().manual_seed(seed)
    datas = []
    for _ in range(num_graphs):
        x = torch.randn(n, INPUT_DIM, generator=g)
        src, dst = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
        ei = torch.tensor([src, dst], dtype=torch.int64)
        datas.append(Data(x=x, edge_index=ei,
                          edge_attr=torch.ones(ei.shape[1], dtype=torch.float32)))
    return Batch.from_data_list(datas)


def _node_weights(batch, keep=KEEP):
    """Hard 0/1 mask: node i of each graph survives iff its within-graph index < keep."""
    w = torch.zeros(batch.x.shape[0], 1)
    for gi in range(int(batch.batch.max()) + 1):
        idx = (batch.batch == gi).nonzero(as_tuple=True)[0]
        w[idx[:keep]] = 1.0
    return w


def _edge_weights_from_nodes(batch, nw):
    """SIR's construction: EdgeWeight(u,v) = NW(u) * NW(v)."""
    src, dst = batch.edge_index
    return (nw.view(-1)[src] * nw.view(-1)[dst]).to(torch.float32)


def _perturb_masked(batch, nw, scale=50.0, seed=1234):
    """Return a copy of `batch` with ONLY the masked-out nodes' features changed."""
    g = torch.Generator().manual_seed(seed)
    b2 = batch.clone()
    masked = (nw.view(-1) == 0)
    b2.x = batch.x.clone()
    b2.x[masked] = torch.randn(int(masked.sum()), INPUT_DIM, generator=g) * scale
    return b2


def _build(pool, norm_mode, attn_renorm, seed=0):
    torch.manual_seed(seed)
    return GNN(input_dim=INPUT_DIM, hidden_dim=HIDDEN, output_dim=OUTPUT, edge_dim=None,
               num_layer=2, layer_name="WeightedGATv2", pool_name=pool, heads=2,
               dropout=0.0, norm_mode=norm_mode, attn_renorm=attn_renorm).eval()


def measure(pool="mean", norm_mode="graph", attn_renorm=False, keep=KEEP):
    """Max abs change in what the policy consumes, when masked node features change.

    Returns (leak_on_consumed_output, leak_on_retained_node_features).
    """
    gnn = _build(pool, norm_mode, attn_renorm)
    b1 = _batch()
    nw = _node_weights(b1, keep)
    b1.edge_attr = _edge_weights_from_nodes(b1, nw)
    b2 = _perturb_masked(b1, nw)
    b2.edge_attr = b1.edge_attr.clone()          # identical mask, identical gating

    with torch.no_grad():
        e1, m1 = gnn(b1, node_weights=nw)
        e2, m2 = gnn(b2, node_weights=nw)

    if m1 is not None:                            # pool_name="none": per-node tokens
        assert torch.equal(m1, m2), "validity mask itself changed"
        # Only the tokens the head may attend to count.
        d_out = (e1[m1] - e2[m2]).abs().max().item()
    else:
        d_out = (e1 - e2).abs().max().item()

    # Hidden features of the RETAINED nodes, before readout.
    feats = {}
    h = gnn.norms[-1].register_forward_hook(
        lambda mod, i, o: feats.__setitem__(len(feats), o.detach().clone()))
    with torch.no_grad():
        gnn(b1, node_weights=nw); gnn(b2, node_weights=nw)
    h.remove()
    kept = (nw.view(-1) > 0)
    d_feat = (feats[0][kept] - feats[1][kept]).abs().max().item()
    return d_out, d_feat


# ----------------------------------------------------------------- the audit grid
def test_shipped_config_leaks():
    """Baseline: the shipped settings DO leak. This test documents the bug."""
    d_out, d_feat = measure(norm_mode="graph", attn_renorm=False)
    assert d_feat > 1e-3, f"expected a leak on retained features, got {d_feat}"
    assert d_out > 1e-3, f"expected a leak on the consumed output, got {d_out}"


def test_norm_mode_node_alone_is_not_enough():
    """Closing LayerNorm leaves the attention-softmax path open."""
    d_out, d_feat = measure(norm_mode="node", attn_renorm=False)
    assert d_out > 1e-6, (
        f"attention path unexpectedly clean ({d_out}); if this now passes, "
        "re-derive the leak-path enumeration")


def test_attn_renorm_alone_is_not_enough():
    """Closing attention leaves the LayerNorm path open."""
    d_out, _ = measure(norm_mode="graph", attn_renorm=True)
    assert d_out > 1e-3, f"LayerNorm path unexpectedly clean ({d_out})"


def test_both_flags_give_zero_contribution():
    """THE GUARANTEE: with both paths closed, pruned nodes contribute nothing."""
    for pool in ("mean", "none", "add", "max"):
        d_out, d_feat = measure(pool=pool, norm_mode="node", attn_renorm=True)
        assert d_feat < 1e-5, f"[pool={pool}] retained features moved by {d_feat}"
        assert d_out < 1e-5, f"[pool={pool}] consumed output moved by {d_out}"


def test_pool_none_masked_tokens_are_excluded():
    """pool_name='none': the head must not even see a token for a pruned node."""
    gnn = _build("none", "node", True)
    b = _batch()
    nw = _node_weights(b)
    b.edge_attr = _edge_weights_from_nodes(b, nw)
    with torch.no_grad():
        emb, mask = gnn(b, node_weights=nw)
    assert mask is not None, "pool_name='none' must return a validity mask"
    # Exactly KEEP tokens per graph are attendable, and they are the kept ones.
    assert mask.sum(dim=1).tolist() == [KEEP] * N_GRAPHS, mask.sum(dim=1).tolist()
    assert emb.shape[0] == N_GRAPHS


def _action_from(gnn, mdt, batch, nw, act, goal, sigma):
    """Full path: graph -> encoder -> per-node tokens -> MDT -> action."""
    with torch.no_grad():
        emb, mask = gnn(batch, node_weights=nw)
        if mask is None:
            emb = emb.unsqueeze(1)
            mask = torch.ones(emb.shape[0], 1, dtype=torch.bool)
        return mdt({"obs_graph": {_MOD: (emb, mask)}}, act, goal, sigma)


def _end_to_end_leak(pool, norm_mode, attn_renorm):
    from networks.action_generator.mdt import MDT
    torch.manual_seed(3)
    gnn = _build(pool, norm_mode, attn_renorm)
    mdt = MDT(sigma_data=0.5, obs_seq_len=1, action_dim=12, proprio_dim=0, img_dim=512,
              graph_dim=OUTPUT, goal_dim=512, img_mod=[], graph_mod=[_MOD],
              use_graph_fusion=True, embed_dim=64, embed_pdrob=0.0, attn_pdrop=0.0,
              resid_pdrop=0.0, mlp_pdrop=0.0, n_dec_layers=2, n_enc_layers=2, n_heads=4,
              max_graph_nodes=32).eval()
    b1 = _batch()
    nw = _node_weights(b1)
    b1.edge_attr = _edge_weights_from_nodes(b1, nw)
    b2 = _perturb_masked(b1, nw); b2.edge_attr = b1.edge_attr.clone()
    torch.manual_seed(11)
    act = torch.randn(N_GRAPHS, 10, 12)
    goal = {"lang": torch.randn(N_GRAPHS, 512)}
    sigma = torch.rand(N_GRAPHS) + 0.1
    a1 = _action_from(gnn, mdt, b1, nw, act, goal, sigma)
    a2 = _action_from(gnn, mdt, b2, nw, act, goal, sigma)
    return (a1 - a2).abs().max().item(), a1.abs().max().item()


def test_action_leaks_on_shipped_config():
    d, scale = _end_to_end_leak("none", "graph", False)
    assert d > 1e-4, f"expected the emitted action to move, got {d} (scale {scale})"


def test_action_is_bit_identical_when_both_paths_closed():
    """THE PAPER CLAIM, end to end: perturbing pruned nodes cannot move the action."""
    for pool in ("none", "mean"):
        d, scale = _end_to_end_leak(pool, "node", True)
        assert d < 1e-5, f"[pool={pool}] action moved by {d} (scale {scale})"


def test_vgib_graph_mean_is_a_third_leak_path():
    """VGIB leak: the noise prior mu_g is the mean over ALL nodes, pruned included.

    _vgib:  mu_g = (x_dense * valid).sum(1) / counts     <- `valid` is padding, not
                                                            the kept mask
            z_mean = p * x + (1 - p) * mu

    So every kept node with p < 1 carries (1-p) * mu_g, and mu_g moves when a pruned
    node's features move. Only p == 1 exactly is clean. Active in eval_prune_mode
    'soft' and 'noise', and in ALL modes during training.

    Documented, not fixed: restricting mu_g to kept nodes changes what VGIB's prior
    means, which is a method change, not a bug fix.
    """
    torch.manual_seed(0)
    n, d = 6, 12
    x = torch.randn(n, d)
    probs = torch.tensor([1.0, 0.9, 0.8, 0.0, 0.0, 0.0])

    def z_mean_of(feats):
        p = probs.unsqueeze(-1)
        return p * feats + (1 - p) * feats.mean(dim=0)

    x2 = x.clone()
    x2[3:] = torch.randn(3, d) * 50.0
    kept = probs > 0
    moved = (z_mean_of(x)[kept] - z_mean_of(x2)[kept]).abs().max().item()
    assert moved > 1.0, f"expected the VGIB prior to leak, got {moved}"
    # ...and the p == 1 node specifically is clean, which is why 'hard' escapes it.
    exact = (z_mean_of(x)[0] - z_mean_of(x2)[0]).abs().max().item()
    assert exact < 1e-6, f"p=1 node should be clean, moved {exact}"


def test_batch_independence():
    """A graph's embedding must not depend on which graphs share its batch.

    mode='graph' without a `batch` argument reduces over the whole batch tensor, so
    batch size 16 in training and 1 at rollout are different functions.
    """
    for norm_mode, should_match in (("graph", False), ("node", True)):
        gnn = _build("mean", norm_mode, attn_renorm=False)
        b_alone = _batch(num_graphs=1)
        b_crowd = Batch.from_data_list(
            b_alone.to_data_list() + _batch(seed=7, num_graphs=3).to_data_list())
        with torch.no_grad():
            e_alone, _ = gnn(b_alone)
            e_crowd, _ = gnn(b_crowd)
        d = (e_alone[0] - e_crowd[0]).abs().max().item()
        if should_match:
            assert d < 1e-6, f"norm_mode={norm_mode}: embedding moved by {d} with batch-mates"
        else:
            assert d > 1e-6, f"norm_mode={norm_mode}: expected batch dependence, got {d}"



def test_fixed_path_does_not_zero_features():
    """The shipped `fixed` branch returns x_out=None, so nothing zeroes the pruned nodes'
    features and attn_renorm is the only thing standing between a fixed-set ablation and
    a leak -- unless true_removal is on, whose branch must return the zeroed features."""
    # Source-level: instantiating a `fixed` module needs a whitelist on disk, and the
    # property under test is structural, not runtime.
    import inspect
    from networks.graph_encoder.gnn import Sparsification_Module
    # The WHOLE CLASS, not one method, so the check survives `forward` being split up.
    src = inspect.getsource(Sparsification_Module)
    import re
    # Match the exact return statements rather than splitting on keywords: the previous
    # version split on "return" and ran past the end of the `fixed` block into the rest of
    # the method, so it was asserting against whatever returned last in forward().
    assert "true_removal" in src, "the fixed branch lost its true-removal path"
    shipped = re.search(
        r"return edge_weights, sampled_nodes, probs, (\w+), probs\.clone\(\)", src)
    removal = re.search(
        r"return ew, sampled_nodes, probs, (\w+), probs\.clone\(\)", src)
    assert shipped and shipped.group(1) == "None", (
        "the SHIPPED fixed path no longer returns x_out=None; if it now zeroes features, "
        "attn_renorm may no longer be load-bearing for fixed-set ablations")
    assert removal and removal.group(1) == "x_out", (
        "the true-removal path does not return zeroed features")
    print("  ok: true-removal path returns x_out; the shipped path still returns None, "
          "so attn_renorm stays load-bearing for un-migrated ablations")


# ---- TRUE REMOVAL under `fixed` ----
def test_fixed_true_removal_equals_a_graph_without_the_nodes():
    """The strongest form of the audit: a pruned node must be indistinguishable from a node
    that was NEVER THERE.

    Weaker checks (perturb the pruned features, see if the output moves) pass on a
    construction that still leaks through the attention denominator, because a zeroed
    feature is constant and a constant leak is invisible to a perturbation test. This
    compares against the ground truth instead: build the same graph WITHOUT the pruned
    nodes at all and require bit-identical output on the survivors.

    Run in TRAINING mode with gradients live, because `fixed` is a training setting -- that
    is the whole point: the shipped path trains with the pruned nodes' real features in the
    kept nodes' softmax, and a test that only ran at eval would not catch it.
    """
    import torch
    from torch_geometric.data import Data, Batch
    from networks.graph_encoder.gnn import GNN

    torch.manual_seed(0)
    dim, hid, out = 12, 32, 16
    n_keep, n_drop = 4, 5
    x_keep = torch.randn(n_keep, dim)
    x_drop = torch.randn(n_drop, dim) * 3.0          # deliberately large and different
    n = n_keep + n_drop

    def full_graph(xk, xd):
        x = torch.cat([xk, xd])
        src, dst = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
        ei = torch.tensor([src, dst], dtype=torch.int64)
        return x, ei

    def small_graph(xk):
        src, dst = zip(*[(i, j) for i in range(n_keep) for j in range(n_keep) if i != j])
        return torch.tensor([src, dst], dtype=torch.int64)

    g = GNN(dim, hid, out, None, 2, "WeightedGATv2", "none", 2, 0.0,
            norm_mode="node", attn_renorm=False)
    g.train()                                        # TRAINING mode, as the arms run

    # --- reference: the graph that never contained the pruned nodes ---
    xk_ref = x_keep.clone().requires_grad_(True)
    ei_s = small_graph(xk_ref)
    b_ref = Batch.from_data_list([Data(x=xk_ref, edge_index=ei_s,
                                       edge_attr=torch.ones(ei_s.shape[1]))])
    out_ref, _ = g(b_ref, node_weights=torch.ones(n_keep))

    # --- true removal: the full graph, pruned down by the fixed path's mechanism ---
    xk = x_keep.clone().requires_grad_(True)
    x_full, ei_f = full_graph(xk, x_drop)
    probs = torch.cat([torch.ones(n_keep), torch.zeros(n_drop)])
    keep = probs > 0
    ek = keep[ei_f[0]] & keep[ei_f[1]]
    b_rm = Batch.from_data_list([Data(x=(x_full * probs.unsqueeze(-1)),
                                      edge_index=ei_f[:, ek],
                                      edge_attr=torch.ones(int(ek.sum())))])
    out_rm, _ = g(b_rm, node_weights=probs)

    d = (out_ref[0, :n_keep] - out_rm[0, :n_keep]).abs().max().item()
    assert d == 0.0, f"true removal is NOT identical to absence: max |delta| = {d:.3e}"
    print(f"  ok: true removal is bit-identical to a graph without the nodes (delta = {d})")

    # Gradients must be live and must match -- a test on a detached graph would pass
    # vacuously while training was still leaking.
    out_ref[0, :n_keep].sum().backward()
    out_rm[0, :n_keep].sum().backward()
    assert xk_ref.grad is not None and xk.grad is not None, "no gradient reached the inputs"
    gd = (xk_ref.grad - xk.grad).abs().max().item()
    assert gd == 0.0, f"gradients differ: max |delta| = {gd:.3e}"
    assert float(xk.grad.abs().sum()) > 0, "gradient is identically zero (test vacuous)"
    print(f"  ok: gradients bit-identical too (delta = {gd}, |grad| = {float(xk.grad.abs().sum()):.3f})")

    # CONTROL: the shipped path must FAIL this, or the test proves nothing.
    xk2 = x_keep.clone().requires_grad_(True)
    x_full2, ei_f2 = full_graph(xk2, x_drop)
    b_old = Batch.from_data_list([Data(x=x_full2, edge_index=ei_f2,
                                       edge_attr=torch.ones(ei_f2.shape[1])
                                                 * probs[ei_f2[0]] * probs[ei_f2[1]])])
    out_old, _ = g(b_old, node_weights=probs)
    d_old = (out_ref[0, :n_keep] - out_old[0, :n_keep]).abs().max().item()
    assert d_old > 1e-3, f"the shipped path matched too -- test is vacuous ({d_old:.3e})"
    print(f"  ok: control -- the SHIPPED fixed path differs by {d_old:.4e} (the leak)")


def test_true_removal_across_every_hard_subset_strategy():
    """TRUE REMOVAL must hold for fixed, topk, topk_PerTask AND threshold.

    topk / topk_PerTask / threshold share the `fixed` construction -- x_out=None, edge
    weights gated AFTER the attention softmax -- so the shipped SIR path trains with the
    pruned nodes' real features reaching the kept nodes: it re-weights, it does not remove.

    Same acceptance criterion as the `fixed` test and for the same reason: compare against
    a graph built WITHOUT the nodes, in training mode with gradients live, with a control
    asserting the shipped path fails it.
    """
    import torch
    from torch_geometric.data import Data, Batch
    from networks.graph_encoder.gnn import GNN

    dim, hid, out = 12, 32, 16
    n_keep, n_drop = 4, 5
    n = n_keep + n_drop
    torch.manual_seed(0)
    x_keep, x_drop = torch.randn(n_keep, dim), torch.randn(n_drop, dim) * 3.0

    def edges(k):
        src, dst = zip(*[(i, j) for i in range(k) for j in range(k) if i != j])
        return torch.tensor([src, dst], dtype=torch.int64)

    g = GNN(dim, hid, out, None, 2, "WeightedGATv2", "none", 2, 0.0,
            norm_mode="node", attn_renorm=False)
    g.train()

    # Reference: the graph that never contained the pruned nodes.
    xk_ref = x_keep.clone().requires_grad_(True)
    ei_s = edges(n_keep)
    out_ref, _ = g(Batch.from_data_list([Data(x=xk_ref, edge_index=ei_s,
                                              edge_attr=torch.ones(ei_s.shape[1]))]),
                   node_weights=torch.ones(n_keep))

    # `probs` as the STE mask these strategies produce: hard 0/1 value, live gradient.
    raw = torch.cat([torch.full((n_keep,), 0.9), torch.full((n_drop,), 0.1)]
                    ).requires_grad_(True)
    hard = (raw > 0.5).to(raw.dtype)
    probs = (hard - raw).detach() + raw          # straight-through, exactly as in gnn.py
    keep = probs != 0

    xk = x_keep.clone().requires_grad_(True)
    x_full = torch.cat([xk, x_drop])
    ei_f = edges(n)
    ek = keep[ei_f[0]] & keep[ei_f[1]]
    out_rm, _ = g(Batch.from_data_list([Data(x=x_full * probs.unsqueeze(-1),
                                             edge_index=ei_f[:, ek],
                                             edge_attr=torch.ones(int(ek.sum())))]),
                  node_weights=probs)

    d = (out_ref[0, :n_keep] - out_rm[0, :n_keep]).abs().max().item()
    assert d < 1e-6, f"true removal != absence on the STE path: max |delta| = {d:.3e}"
    print(f"  ok: STE-mask true removal == a graph without the nodes (delta {d:.1e})")

    out_rm[0, :n_keep].sum().backward(retain_graph=True)
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0, (
        "no gradient reached the SELECTION scores -- true removal broke learnability")
    print(f"  ok: gradient still reaches the selection scores "
          f"(|d/draw| = {float(raw.grad.abs().sum()):.4f})")

    # Control: the shipped construction must FAIL, or this proves nothing.
    xk2 = x_keep.clone().requires_grad_(True)
    x_full2 = torch.cat([xk2, x_drop])
    out_old, _ = g(Batch.from_data_list(
        [Data(x=x_full2, edge_index=ei_f,
              edge_attr=torch.ones(ei_f.shape[1]) * probs[ei_f[0]].detach()
                        * probs[ei_f[1]].detach())]), node_weights=probs.detach())
    d_old = (out_ref[0, :n_keep] - out_old[0, :n_keep]).abs().max().item()
    assert d_old > 1e-3, f"the shipped path matched too -- test vacuous ({d_old:.3e})"
    print(f"  ok: control -- the SHIPPED topk/threshold path differs by {d_old:.4e}")


def test_true_removal_flag_reaches_every_hard_subset_strategy():
    """Source-level: the flag must be consulted on BOTH branches (fixed, and
    topk/topk_PerTask/threshold)."""
    import re
    from networks.graph_encoder.gnn import Sparsification_Module
    src = inspect_source = __import__("inspect").getsource(Sparsification_Module)
    n = len(re.findall(r"if self\.true_removal:", src))
    assert n >= 2, (
        f"true_removal is consulted on {n} branch(es); it must guard the `fixed` path AND "
        "the topk/topk_PerTask/threshold path")
    print(f"  ok: true_removal is consulted on {n} branches (fixed and topk/threshold)")


if __name__ == "__main__":
    print("=" * 72)
    print("MEASURED LEAK (max abs change from perturbing ONLY masked-node features)")
    print("=" * 72)
    print(f"{'norm_mode':>10} {'attn_renorm':>12} | {'consumed output':>16} {'retained feats':>15}")
    for nm in ("graph", "node"):
        for ar in (False, True):
            d_out, d_feat = measure(norm_mode=nm, attn_renorm=ar)
            print(f"{nm:>10} {str(ar):>12} | {d_out:>16.3e} {d_feat:>15.3e}")
    print()

    tests = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = []
    for name in tests:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed.append((name, str(e))); print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}")); print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    sys.exit(0 if not failed else 1)
