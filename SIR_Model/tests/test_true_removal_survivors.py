"""`true_removal` must remove the dropped AND leave the survivors alone.

THE DEFECT THIS GUARDS. The `topk`/`topk_PerTask`/`threshold` selection block ends in

    indexes = ((probs > thr).to(probs.dtype) - probs).detach() + probs
    probs   = probs * indexes

`indexes` has VALUE 1 for a kept node, so a DROPPED node leaves at 0 -- correct -- and a
KEPT node leaves at **p, its own sigmoid score**. `x_out = x * probs` would then attenuate
every survivor in proportion to how confident the scorer happened to be, and
`Multi_XAI_GNN` passes the same `probs` as the readout's node weights under
`multiplicative_pooling`, so the attenuation would land twice. That is removal of the
dropped AND re-weighting of the kept, which is not what a graph built without the pruned
nodes looks like.

`fixed` cannot show this -- `_whitelist_probs` writes literal 1.0 -- so the soft-scored
strategies are tested here explicitly, against a graph CONSTRUCTED without the nodes.

Run: python tests/test_true_removal_survivors.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch_geometric.data import Data

from networks.graph_encoder.gnn import Sparsification_Module

IN, HID, ED = 57, 32, 1
TASK = "CloseDrawer"
NAMES = ["PandaMobile", "drawer", "counter", "stove", "toaster", "sink"]
N = len(NAMES)
K = 3


def _graph(names, x, ei, ew):
    g = Data(x=x, edge_index=ei, edge_attr=ew)
    g.node_names = [list(names)]
    g.batch = torch.zeros(x.shape[0], dtype=torch.long)
    g.num_graphs = 1
    return g


def _big():
    gen = torch.Generator().manual_seed(17)
    x = torch.rand(N, IN, generator=gen)
    ei = torch.stack([torch.arange(N).repeat_interleave(N), torch.arange(N).repeat(N)])
    return _graph(NAMES, x, ei, torch.rand(ei.shape[1], generator=gen))


def _without(big, keep_idx):
    """A graph CONSTRUCTED from the survivors alone: no row, name or edge for the rest."""
    remap = {o: n for n, o in enumerate(keep_idx)}
    ei = big.edge_index.tolist()
    ek = [e for e in range(len(ei[0])) if ei[0][e] in remap and ei[1][e] in remap]
    sel = torch.tensor(keep_idx, dtype=torch.long)
    return _graph([NAMES[i] for i in keep_idx], big.x[sel],
                  torch.tensor([[remap[ei[0][e]] for e in ek],
                                [remap[ei[1][e]] for e in ek]], dtype=torch.long),
                  big.edge_attr[torch.tensor(ek, dtype=torch.long)])


def _module(**kw):
    kw.setdefault("threshold", 0.5)
    m = Sparsification_Module(in_dim=IN, hidden_dim=HID, num_heads=2, num_layers=1,
                              dropout_prob=0.0, **kw)
    m.eval()
    return m


def _run(m, g):
    with torch.no_grad():
        ew, sampled, probs, x_out, raw = m(
            g.x, g.edge_index, edge_attr=None, edge_weights=g.edge_attr,
            lang_emb=torch.zeros(1, HID), batch=g, task_names=[TASK])
    ek = m._edge_keep
    ei = g.edge_index if ek is None else g.edge_index[:, ek]
    return dict(probs=probs, x=x_out, sampled=sorted(sampled.tolist()),
                edges={(int(a), int(b)) for a, b in zip(*ei.tolist())})


def _case(strategy, **extra):
    """(treatment on the big graph, reference on the graph without the dropped nodes)."""
    big = _big()
    torch.manual_seed(5)
    m = _module(sampling_strategy=strategy, true_removal=True, **extra)
    a = _run(m, big)
    keep_idx = a["sampled"]
    small = _without(big, keep_idx)
    # The reference keeps every node of the smaller graph at EXACTLY 1.0 -- which is what
    # `fixed` emits, and what the treatment must now also emit for a survivor.
    torch.manual_seed(5)
    ref = _module(sampling_strategy="fixed", true_removal=True,
                  node_whitelist={TASK: set(small.node_names[0])})
    b = _run(ref, small)
    return big, small, keep_idx, a, b


def test_survivors_are_not_attenuated():
    for strategy, extra in (("topk", dict(sample_abs=K)),
                            ("topk_PerTask", dict()),
                            ("threshold", dict(threshold=0.77))):
        big, small, keep, a, b = _case(strategy, **extra)
        assert keep, f"{strategy}: nothing selected -- the case is vacuous"
        assert len(keep) < N, f"{strategy}: everything selected -- the case is vacuous"
        for i in keep:
            assert abs(float(a["probs"][i]) - 1.0) < 1e-6, \
                f"{strategy}: survivor {i} left at p={float(a['probs'][i]):.4f}, not 1.0"
        sel = torch.tensor(keep, dtype=torch.long)
        d = float((a["x"][sel] - big.x[sel]).abs().max())
        assert d == 0.0, f"{strategy}: survivor features attenuated by {d:.4e}"
        print(f"  ok: {strategy:14s} {len(keep)}/{N} kept, survivors at 1.0, features intact")


def test_equals_a_graph_built_without_the_pruned_nodes():
    """The §28 criterion itself."""
    for strategy, extra in (("topk", dict(sample_abs=K)),
                            ("topk_PerTask", dict()),
                            ("threshold", dict(threshold=0.77))):
        big, small, keep, a, b = _case(strategy, **extra)
        assert 0 < len(keep) < N, (f"{strategy}: {len(keep)}/{N} kept -- a case that "
                                   f"prunes nothing cannot detect a leak")
        sel = torch.tensor(keep, dtype=torch.long)
        dp = float((a["probs"][sel] - b["probs"]).abs().max())
        dx = float((a["x"][sel] - b["x"]).abs().max())
        remap = {o: n for n, o in enumerate(keep)}
        mapped = {(remap[i], remap[j]) for i, j in a["edges"]}
        assert mapped == b["edges"], f"{strategy}: surviving edge sets differ"
        assert max(dp, dx) == 0.0, \
            f"{strategy}: differs from a constructed graph by {max(dp, dx):.4e}"
        print(f"  ok: {strategy:14s} identical to a graph built without the pruned nodes")


def test_the_attenuating_path_is_a_failing_control():
    """The attenuating behaviour, re-run here so the comparison cannot go vacuous."""
    big = _big()
    torch.manual_seed(5)
    m = _module(sampling_strategy="topk", sample_abs=K, true_removal=True)
    a = _run(m, big)
    keep = a["sampled"]
    small = _without(big, keep)
    torch.manual_seed(5)
    ref = _module(sampling_strategy="fixed", true_removal=True,
                  node_whitelist={TASK: set(small.node_names[0])})
    b = _run(ref, small)

    # verbatim re-implementation of the pre-fix line: probs stays at p * hard
    torch.manual_seed(5)
    old = _module(sampling_strategy="topk", sample_abs=K, true_removal=False)
    with torch.no_grad():
        _, _, p_soft, _, _ = old(big.x, big.edge_index, edge_attr=None,
                                 edge_weights=big.edge_attr,
                                 lang_emb=torch.zeros(1, HID), batch=big,
                                 task_names=[TASK])
    sel = torch.tensor(keep, dtype=torch.long)
    x_old = big.x * p_soft.unsqueeze(-1)          # what x_out used to be
    d = max(float((p_soft[sel] - b["probs"]).abs().max()),
            float((x_old[sel] - b["x"]).abs().max()))
    assert d > 1e-3, ("the attenuating path now agrees with a constructed graph -- the "
                      "fixture no longer exercises the defect")
    print(f"  ok: the shipped path attenuates survivors by {d:.4e}")
    print(f"      survivor scores were {[round(float(p_soft[i]), 4) for i in keep]}, "
          f"not 1.0")


def test_fixed_path_was_never_affected():
    """`fixed` emits literal 1.0 for survivors, so it was never attenuated."""
    big = _big()
    torch.manual_seed(5)
    m = _module(sampling_strategy="fixed", true_removal=True,
                node_whitelist={TASK: set(NAMES[:K])})
    a = _run(m, big)
    for i in a["sampled"]:
        assert float(a["probs"][i]) == 1.0, f"fixed: survivor {i} not at 1.0"
    print("  ok: `fixed` writes literal 1.0 for every survivor")


def test_multi_xai_gnn_slices_the_edges():
    """End to end through the wrapper: `Multi_XAI_GNN` must apply `_edge_keep` to the
    batch, so its readout equals the same GNN run on a graph built without the nodes.

    The module-level tests above check the mask; this checks that the consumer uses it.
    """
    from omegaconf import OmegaConf
    from torch_geometric.data import Batch
    from networks.graph_encoder.gnn import Multi_XAI_GNN

    mod = "m"
    keep_names = NAMES[:K]
    spars_cfg = OmegaConf.create({
        "_target_": "networks.graph_encoder.gnn.Sparsification_Module",
        "hidden_dim": HID, "num_heads": 2, "num_layers": 1, "dropout_prob": 0.0,
        "sampling_strategy": "fixed", "true_removal": True,
        "node_whitelist": {TASK: list(keep_names)}})
    torch.manual_seed(0)
    enc = Multi_XAI_GNN(spars_cfg, {mod: IN}, HID, 16, None, 2, "WeightedGATv2", "mean",
                        2, 0.0, [mod], multiplicative_pooling=True,
                        norm_mode="node").eval()

    def batch_of(g):
        d = Data(x=g.x.clone(), edge_index=g.edge_index.clone(),
                 edge_attr=g.edge_attr.clone())
        d.node_names = g.node_names[0]
        return Batch.from_data_list([d])

    # Real scene graphs carry no self-loops (utils/create_graphs.py skips obj == sub), and
    # WeightedGATv2Conv's edge-weight alignment assumes none, so drop them here.
    big = _big()
    no_sl = big.edge_index[0] != big.edge_index[1]
    big = _graph(NAMES, big.x, big.edge_index[:, no_sl], big.edge_attr[no_sl])
    small = _without(big, list(range(K)))
    with torch.no_grad():
        out_big, _ = enc({mod: batch_of(big)}, lang_emb=torch.zeros(1, HID),
                         task_names=[TASK])[mod]
        out_small, _ = enc.gnn_models[mod](batch_of(small), node_weights=torch.ones(K))
    d = float((out_big - out_small).abs().max())
    assert d < 1e-5, f"Multi_XAI_GNN true removal differs from a graph without the nodes by {d:.3e}"
    ratio = enc.last_sparsification[mod]["kept_edges_ratio"]
    want = (K * (K - 1)) / (N * (N - 1))
    assert abs(ratio - want) < 1e-9, f"kept_edges_ratio {ratio} != {want}"
    print(f"  ok: wrapper output == graph without the nodes (delta {d:.1e}), "
          f"kept_edges_ratio {ratio:.3f}")


TESTS = (test_survivors_are_not_attenuated,
         test_equals_a_graph_built_without_the_pruned_nodes,
         test_the_attenuating_path_is_a_failing_control,
         test_fixed_path_was_never_affected,
         test_multi_xai_gnn_slices_the_edges)

if __name__ == "__main__":
    fails = 0
    for fn in TESTS:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} tests passed")
    sys.exit(1 if fails else 0)
