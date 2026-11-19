"""Every readout must normalise over the SURVIVORS, and this test enumerates them itself.

THE RULE. Any operation that normalises or reduces over a set of nodes -- softmax,
LayerNorm, mean, max -- must be computed over the set that REMAINS after removal. Zeroing
a node is not removing it: a normaliser that still sees the removed nodes turns every
ablation into a magnitude change, so the effect of removing nodes then measures how many
were removed rather than what they contained.

IT ENUMERATES POOL MODES FROM THE SOURCE rather than from a hand-written list, so adding
a `pool_name` without a removal-equivalence story fails here instead of shipping. The
comparison is always against a graph CONSTRUCTED WITHOUT the removed nodes -- never
against a perturbed graph, because a constant leak is invisible to a perturbation.

Run: python tests/test_normalisation_over_survivors.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch_geometric.data import Batch, Data

from networks.graph_encoder.gnn import GNN

IN, HID, OUT, ED = 16, 32, 24, 4
KEEP, DROP = 3, 5
EPS = 1e-5          # float32 slack: the two runs reduce different token counts
GNN_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "networks", "graph_encoder", "gnn.py")


def supported_pool_names():
    """Parse the pool_name branches out of GNN.__init__ -- no hand-written list."""
    src = open(GNN_SRC).read()
    names = re.findall(r'pool_name == "(\w+)"', src)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def _graphs(seed=1):
    torch.manual_seed(seed)
    x = torch.randn(KEEP + DROP, IN)
    kei = torch.stack([torch.arange(KEEP).repeat_interleave(KEEP),
                       torch.arange(KEEP).repeat(KEEP)])
    ea = torch.randn(KEEP * KEEP, ED)
    # Ablated view: every node present, edges only among survivors -- what
    # true_removal produces once the consumer has sliced the edges.
    abl = Data(x=x.clone(), edge_index=kei, edge_attr=ea.clone())
    ref = Data(x=x[:KEEP].clone(), edge_index=kei, edge_attr=ea.clone())
    return abl, ref


def _gnn(pool, norm_mode="node"):
    torch.manual_seed(0)
    return GNN(IN, HID, OUT, ED, 2, "GATv2", pool, heads=4, dropout=0.0,
               norm_mode=norm_mode).eval()


def _run(pool, abl, ref, norm_mode="node", tell_pool=True):
    g = _gnn(pool, norm_mode)
    w = torch.cat([torch.ones(KEEP), torch.zeros(DROP)]) if tell_pool else None
    with torch.no_grad():
        a, am = g(Batch.from_data_list([abl]), node_weights=w)
        r, rm = g(Batch.from_data_list([ref]), node_weights=torch.ones(KEEP))
    if a.dim() == 3:                      # pool_name="none": per-node readout
        a, r = a[:, :KEEP, :], r[:, :KEEP, :]
        assert am is not None and rm is not None
        assert int(am[0, :KEEP].sum()) == KEEP == int(rm[0, :KEEP].sum()), \
            "survivor tokens are not all attendable"
        assert int(am[0, KEEP:].sum()) == 0, "removed nodes stayed attendable"
    return float((a - r).abs().max())


def test_every_pool_mode_normalises_over_survivors():
    modes = supported_pool_names()
    print(f"  pool modes found in GNN.__init__: {modes}")
    assert modes, "parsed no pool_name branches -- the enumeration broke, not the code"
    abl, ref = _graphs()
    bad = []
    for m in modes:
        d = _run(m, abl, ref)
        ok = d < EPS
        print(f"    pool_name={m:<6} max|ablated - graph-without-nodes| = {d:.2e} "
              f"{'ok' if ok else 'LEAKS'}")
        if not ok:
            bad.append((m, d))
    assert not bad, (
        f"pool modes whose readout does not normalise over the survivors: {bad}. "
        "Removed nodes must leave the normaliser (softmax/mean/max/LayerNorm), not merely "
        "be zeroed.")


def test_max_pool_negative_survivors():
    """Zero must not win a max. The generic fixture only exercises channels where every
    survivor happens to be negative; this forces ALL of them negative.

    The last LayerNorm's affine is pinned to weight 0 / bias -1, so every node leaves the
    encoder at leaky_relu(-1) = -0.01 in every channel. A removed node is zeroed by the
    node weights, so a max that still sees it returns 0 instead of -0.01.
    """
    abl, ref = _graphs()
    g = _gnn("max")
    with torch.no_grad():
        g.norms[-1].weight.fill_(0.0)
        g.norms[-1].bias.fill_(-1.0)
        w = torch.cat([torch.ones(KEEP), torch.zeros(DROP)])
        a, _ = g(Batch.from_data_list([abl]), node_weights=w)
        r, _ = g(Batch.from_data_list([ref]), node_weights=torch.ones(KEEP))
        survivor = g.output(torch.full((1, HID), -0.01))
        zero_won = g.output(torch.zeros(1, HID))
    assert float((a - r).abs().max()) < EPS, "max pool differs from a graph without the nodes"
    assert float((a - survivor).abs().max()) < EPS, "max pool did not return the survivors' value"
    assert float((zero_won - survivor).abs().max()) > EPS, \
        "control: a zero-won max is indistinguishable here -- fixture is vacuous"
    print("  ok: max pool returns the survivors' -0.01, not the removed nodes' 0")


def test_the_check_can_fail():
    """Controls: the comparison is sensitive enough to catch the real defects.

    1. Leave the removed nodes in the readout (no node_weights), so the mean counts them.
    2. Graph-mode LayerNorm, whose statistics are pooled over every node present.
    Both must be caught, or the passes above mean nothing.
    """
    abl, ref = _graphs()
    d_pool = _run("mean", abl, ref, tell_pool=False)
    d_norm = _run("mean", abl, ref, norm_mode="graph")
    assert d_pool > EPS, (f"control 1 did not leak ({d_pool:.2e}); this test can no "
                          "longer detect removed nodes left in the readout")
    assert d_norm > EPS, (f"control 2 did not leak ({d_norm:.2e}); this test can no "
                          "longer detect a normaliser that sees removed nodes")
    print(f"  ok: controls -- removed nodes in the mean differ by {d_pool:.2e}, "
          f"graph-mode LayerNorm by {d_norm:.2e}")


TESTS = (test_every_pool_mode_normalises_over_survivors,
         test_max_pool_negative_survivors,
         test_the_check_can_fail)

if __name__ == "__main__":
    fails = 0
    for fn in TESTS:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} tests passed")
    sys.exit(1 if fails else 0)
