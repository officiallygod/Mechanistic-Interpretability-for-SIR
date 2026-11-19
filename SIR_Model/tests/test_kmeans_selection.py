"""Tests for the unsupervised per-task subgraph cut (utils.select_subgraph --kmeans).

This is the only thing standing between a stage-1 score distribution and the subgraph
that gets pruned at eval and distilled in stage 2, and it is silent when it goes wrong:
a bad split still produces a well-formed YAML and a run that trains. Two failures matter
most -- an EMPTY subgraph makes the policy scene-blind, and a threshold outside the gap
it claims to sit in flips nodes as soon as eval re-scores them.

Run: python tests/test_kmeans_selection.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.select_subgraph import kmeans_split
from utils.node_scores import load_node_whitelist


def _ranked(pairs):
    return sorted(pairs, key=lambda kv: -kv[1])


def test_clear_gap_is_cut_at_the_gap():
    keep, thr = kmeans_split(_ranked([("a", 0.95), ("b", 0.90), ("c", 0.88),
                                      ("d", 0.50), ("e", 0.48), ("f", 0.45)]))
    assert set(keep) == {"a", "b", "c"}, keep
    assert 0.50 < thr < 0.88, thr


def test_threshold_sits_strictly_inside_the_gap():
    # The bar has to be BETWEEN the lowest kept and the highest dropped score, or eval's
    # freshly predicted scores land on the wrong side of it.
    scores = [("a", 0.9), ("b", 0.85), ("c", 0.4), ("d", 0.3)]
    keep, thr = kmeans_split(_ranked(scores))
    kept_vals = [v for n, v in scores if n in keep]
    drop_vals = [v for n, v in scores if n not in keep]
    assert min(kept_vals) > thr > max(drop_vals), (min(kept_vals), thr, max(drop_vals))


def test_never_returns_an_empty_subgraph():
    # A scene-blind policy is the one outcome we can never ship, so check the degenerate
    # inputs: flat scores, two nodes, one node.
    for scores in ([("a", 0.5), ("b", 0.5), ("c", 0.5), ("d", 0.5)],
                   [("a", 0.7), ("b", 0.2)],
                   [("a", 0.7)]):
        keep, thr = kmeans_split(_ranked(scores))
        assert len(keep) >= 1, (scores, keep)
        assert 0.0 <= thr <= 1.0, thr


def test_flat_scores_keep_everything_rather_than_inventing_a_split():
    keep, thr = kmeans_split(_ranked([(c, 0.6) for c in "abcdef"]))
    assert len(keep) == 6, keep
    assert thr == 0.0, thr


def test_cut_is_a_prefix_of_the_ranking():
    # Whatever it keeps must be the top-k by score: a threshold cannot express any other
    # set, so a non-prefix result would mean the whitelist and the bar disagree.
    scores = [("a", 0.99), ("b", 0.8), ("c", 0.79), ("d", 0.77), ("e", 0.3)]
    ranked = _ranked(scores)
    keep, thr = kmeans_split(ranked)
    names = [n for n, _ in ranked]
    assert keep == names[:len(keep)], (keep, names)
    assert keep == [n for n, v in ranked if v > thr], (keep, thr)


def test_scale_invariance_of_the_partition():
    lo = _ranked([("a", 0.30), ("b", 0.28), ("c", 0.10), ("d", 0.08)])
    hi = _ranked([("a", 0.95), ("b", 0.93), ("c", 0.75), ("d", 0.73)])
    assert set(kmeans_split(lo)[0]) == set(kmeans_split(hi)[0])


def test_output_is_consumable_as_a_stage2_whitelist():
    # The same file feeds eval (task_thresholds) and stage 2 (node_whitelist); if the
    # shapes ever diverge, stage 2 dies on a KeyError deep into a run.
    keep, thr = kmeans_split(_ranked([("PandaMobile", 0.99), ("Drawer", 0.82),
                                      ("Counter", 0.55), ("Floor", 0.48)]))
    doc = {"tasks": {"CloseDrawer": keep}, "task_thresholds": {"CloseDrawer": thr}}
    wl = load_node_whitelist(doc)
    assert wl["CloseDrawer"] == set(keep), wl


def test_relevance_whitelist_reaches_BOTH_lookups():
    """The discovered list must replace RELEVENT_NODES in both places it is consulted.

    These live on DIFFERENT classes -- `k` on Sparsification_Module, the CBM target on
    Multi_XAI_GNN -- and configuring only the first is exactly the bug that killed the
    first idea-6 run 8 minutes into training: `k` was overridden, the CBM loss raised
    AttributeError on the first batch. A config dry-run cannot catch it; only touching
    the attribute from the parent can.
    """
    from omegaconf import OmegaConf
    from networks.graph_encoder.gnn import Multi_XAI_GNN

    wl = {"CloseDrawer": ["PandaMobile", "Drawer"]}
    enc = Multi_XAI_GNN(
        input_dim={"m": 8}, hidden_dim=16, output_dim=16, edge_dim=None, num_layer=1,
        layer_name="WeightedGATv2", pool_name="mean", heads=1, dropout=0.0,
        modalities=["m"], use_cbm_loss="nodes_att", lcbm_weight=0.1,
        sparsification_layer=OmegaConf.create({
            "_target_": "networks.graph_encoder.gnn.Sparsification_Module",
            "in_dim": {}, "hidden_dim": 16, "num_heads": 1, "num_layers": 1,
            "dropout_prob": 0.0, "sampling_strategy": "topk_PerTask",
            "relevance_whitelist": wl,
        }),
    )
    # the parent must see it (CBM target)
    assert enc.relevance_whitelist is not None, "Multi_XAI_GNN cannot see relevance_whitelist"
    assert enc.relevance_whitelist["CloseDrawer"] == {"PandaMobile", "Drawer"}
    # the child must see it (k per task)
    spars = enc.sparsification_modules["m"]
    assert spars._calc_k_for_topk(20, "CloseDrawer") == 2, "k did not come from the whitelist"


def test_no_whitelist_leaves_the_relevance_table_in_charge():
    from omegaconf import OmegaConf
    from networks.graph_encoder.gnn import Multi_XAI_GNN
    enc = Multi_XAI_GNN(
        input_dim={"m": 8}, hidden_dim=16, output_dim=16, edge_dim=None, num_layer=1,
        layer_name="WeightedGATv2", pool_name="mean", heads=1, dropout=0.0,
        modalities=["m"], use_cbm_loss="nodes_att", lcbm_weight=0.1,
        sparsification_layer=OmegaConf.create({
            "_target_": "networks.graph_encoder.gnn.Sparsification_Module",
            "in_dim": {}, "hidden_dim": 16, "num_heads": 1, "num_layers": 1,
            "dropout_prob": 0.0, "sampling_strategy": "topk_PerTask",
        }),
    )
    assert enc.relevance_whitelist is None


if __name__ == "__main__":
    tests = [n for n in dir() if n.startswith("test_")]
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
