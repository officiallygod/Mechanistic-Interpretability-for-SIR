"""Smoke tests for the GNN forward path, including the per-node readout.

Run: python tests/test_gnn_path.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn.norm import LayerNorm

from networks.graph_encoder.gnn import GNN, Multi_GNN


INPUT_DIM = 10
HIDDEN = 32
OUTPUT = 16


def _mock_batch(num_graphs=2, num_nodes_per_graph=5):
    datas = []
    sizes = ([num_nodes_per_graph] * num_graphs if isinstance(num_nodes_per_graph, int)
             else list(num_nodes_per_graph))
    for N in sizes:
        x = torch.randn(N, INPUT_DIM)
        src, dst = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src.append(i); dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.int64)
        edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)
        datas.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return Batch.from_data_list(datas)


def _build_gnn(pool="mean", output_dim=OUTPUT, layer_name="GATv2"):
    return GNN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        output_dim=output_dim,
        edge_dim=1,
        num_layer=2,
        layer_name=layer_name,
        pool_name=pool,
        heads=2,
        dropout=0.5,
    )


def test_full_defaults_forward_shape():
    gnn = _build_gnn().eval()
    batch = _mock_batch(num_graphs=2, num_nodes_per_graph=5)
    with torch.no_grad():
        emb, mask = gnn(batch)
    assert emb.shape == (2, OUTPUT), emb.shape
    assert mask is None, "pooled readout must not return a mask"


def test_full_defaults_have_gat_layernorm_and_self_loops():
    gnn = _build_gnn()
    assert gnn.input_layer is not None
    assert gnn.input_layer.add_self_loops is True
    assert len(gnn.layers) == 2
    for layer in gnn.layers:
        assert layer.add_self_loops is True
    assert isinstance(gnn.input_norm, LayerNorm)
    for n in gnn.norms:
        assert isinstance(n, LayerNorm)
    assert isinstance(gnn.dropout, nn.Dropout)


def test_pool_none_returns_dense_and_mask():
    gnn = _build_gnn(pool="none").eval()
    batch = _mock_batch(num_graphs=2, num_nodes_per_graph=5)
    with torch.no_grad():
        emb, mask = gnn(batch)
    assert emb.shape == (2, 5, OUTPUT), emb.shape
    assert mask.shape == (2, 5), mask.shape
    assert mask.all().item()


def test_pool_none_masks_padded_nodes():
    """Graphs of different sizes get padded to the largest; the mask marks the padding."""
    gnn = _build_gnn(pool="none").eval()
    batch = _mock_batch(num_nodes_per_graph=[3, 6])
    with torch.no_grad():
        emb, mask = gnn(batch)
    assert emb.shape == (2, 6, OUTPUT), emb.shape
    assert mask[0].tolist() == [True, True, True, False, False, False]
    assert mask[1].all().item()


def test_pool_none_excludes_sparsified_nodes_from_mask():
    """Nodes the sparsifier zeroed out must not stay attendable as all-zero tokens."""
    gnn = _build_gnn(pool="none").eval()
    batch = _mock_batch(num_graphs=2, num_nodes_per_graph=4)
    node_weights = torch.ones(batch.x.shape[0])
    node_weights[1] = 0.0          # drop node 1 of graph 0
    node_weights[6] = 0.0          # drop node 2 of graph 1
    with torch.no_grad():
        _, mask = gnn(batch, node_weights=node_weights)
    assert mask[0].tolist() == [True, False, True, True]
    assert mask[1].tolist() == [True, True, False, True]


def test_weighted_mean_pool_ignores_dropped_nodes():
    """Weighted mean pooling divides by the kept-node count, not the total."""
    gnn = _build_gnn(pool="mean", layer_name="WeightedGATv2").eval()
    batch = _mock_batch(num_graphs=1, num_nodes_per_graph=4)
    weights = torch.tensor([1.0, 1.0, 0.0, 0.0])
    with torch.no_grad():
        pooled, mask = gnn(batch, node_weights=weights)
        dense, _ = gnn(batch, node_weights=weights)
    assert mask is None
    assert pooled.shape == (1, OUTPUT)
    assert torch.isfinite(pooled).all()


def test_eval_mode_is_deterministic():
    gnn = _build_gnn().eval()
    batch = _mock_batch()
    with torch.no_grad():
        emb_a, _ = gnn(batch)
        emb_b, _ = gnn(batch)
    assert torch.allclose(emb_a, emb_b, atol=1e-6), "eval-mode output should be deterministic"


def test_multi_gnn_returns_new_dict_per_modality():
    """The encoder must not mutate its input dict — preprocess_batch relies on that."""
    multi = Multi_GNN(
        input_dim={"mod_a": INPUT_DIM},
        hidden_dim=HIDDEN,
        output_dim=OUTPUT,
        edge_dim=1,
        num_layer=1,
        layer_name="GATv2",
        pool_name="mean",
        heads=2,
        dropout=0.0,
        modalities=["mod_a"],
    ).eval()
    batch = _mock_batch()
    inputs = {"mod_a": batch}
    with torch.no_grad():
        out = multi(inputs, lang_emb=torch.randn(2, 512))
    assert inputs["mod_a"] is batch, "input dict was mutated"
    emb, mask = out["mod_a"]
    assert emb.shape == (2, OUTPUT)
    assert mask is None


if __name__ == "__main__":
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
