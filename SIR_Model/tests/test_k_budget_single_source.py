"""The per-task k budget has ONE source, and `relevance_whitelist` moves it.

`_calc_k_for_topk` honoured `relevance_whitelist`, but the live selection path called
`get_num_relevant_nodes_per_task` directly. A run that set the whitelist (e.g.
scripts/pipeline_idea6.sh) therefore trained the CBM against a DISCOVERED list while the
top-k budget stayed on the HAND-WRITTEN one -- confounding "is the discovered list any
good?" with "is the hand-written budget any good?", which is the one thing such a run
exists to separate. Both readers agreed whenever no whitelist was set, which is why it
went unnoticed.

Run: python tests/test_k_budget_single_source.py
"""
import os
import sys

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.graph_encoder.gnn import Sparsification_Module            # noqa: E402
from utils.generate_graph_dataset_robocasa import (                     # noqa: E402
    get_num_relevant_nodes_per_task)

TASK = "CloseDrawer"
IN, HID = 57, 32


def _module(**kw):
    return Sparsification_Module(in_dim=IN, hidden_dim=HID, num_heads=2, num_layers=1,
                                 dropout_prob=0.0, sampling_strategy="topk_PerTask", **kw)


def test_no_whitelist_still_reads_the_table():
    m = _module()
    assert m._k_per_task(TASK) == get_num_relevant_nodes_per_task(TASK)


def test_whitelist_moves_the_budget():
    """The property that was false: k must follow the whitelist, not the table."""
    table_k = get_num_relevant_nodes_per_task(TASK)
    wl_k = table_k + 3
    m = _module(relevance_whitelist={TASK: [f"n{i}" for i in range(wl_k)]})
    assert m._k_per_task(TASK) == wl_k, (
        f"budget fell back to the hand-written table ({m._k_per_task(TASK)}) instead of "
        f"following the whitelist ({wl_k})")


def test_both_callers_agree():
    """`_calc_k_for_topk` caps at the graph size; the selection path applies the same cap
    via `torch.minimum(k_budget, counts)`. With the cap on both sides they must agree."""
    wl_k = get_num_relevant_nodes_per_task(TASK) + 3
    m = _module(relevance_whitelist={TASK: [f"n{i}" for i in range(wl_k)]})
    for n_nodes in (2, wl_k - 1, wl_k, wl_k + 5):
        via_hist = m._calc_k_for_topk(n_nodes, task_name=TASK)
        via_sel = min(m._k_per_task(TASK), n_nodes)
        assert via_hist == via_sel, (n_nodes, via_hist, via_sel)


def test_selection_keeps_the_whitelist_count():
    """End to end through forward(): the number of nodes actually kept follows the
    whitelist, for budgets both above and below the hand-written table's."""
    table_k = get_num_relevant_nodes_per_task(TASK)
    n = table_k + 8
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(n, IN, generator=gen)
    src, dst = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    g = Data(x=x, edge_index=torch.tensor([src, dst]),
             edge_attr=torch.ones(len(src)))
    g.batch = torch.zeros(n, dtype=torch.long)
    g.num_graphs = 1
    for wl_k in (max(1, table_k - 1), table_k + 3):
        m = _module(relevance_whitelist={TASK: [f"n{i}" for i in range(wl_k)]}).eval()
        with torch.no_grad():
            _, sampled, _, _, _ = m(g.x, g.edge_index, edge_attr=None,
                                    edge_weights=g.edge_attr, lang_emb=torch.zeros(1, HID),
                                    batch=g, task_names=[TASK])
        assert sampled.numel() == wl_k, (
            f"kept {sampled.numel()} nodes; whitelist says {wl_k}, table says {table_k}")


def test_missing_task_is_an_error_not_a_silent_fallback():
    m = _module(relevance_whitelist={"SomeOtherTask": ["a"]})
    try:
        m._k_per_task(TASK)
    except KeyError:
        return
    raise AssertionError("a whitelist with no entry for the task silently fell back to "
                         "the hand-written table; that is the defect in another form")


if __name__ == "__main__":
    tests = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = []
    for name in tests:
        try:
            globals()[name](); print(f"PASS  {name}")
        except AssertionError as e:
            failed.append(name); print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append(name); print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    sys.exit(0 if not failed else 1)
