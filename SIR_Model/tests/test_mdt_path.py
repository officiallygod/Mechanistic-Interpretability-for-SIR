"""Smoke tests for the MDT state-token path: masking, sequence sizing, cross attention.

Run: python tests/test_mdt_path.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from networks.action_generator.mdt import MDT
from networks.transformer.transformer_decoders import TransformerFiLMDecoder


MOD = "cropped_image_feature_bb_coordinates"
B, ACT_WINDOW, ACTION_DIM = 3, 10, 12


def _build(graph_mod=(MOD,), max_graph_nodes=32, embed_dim=64):
    return MDT(
        sigma_data=0.5, obs_seq_len=1, action_dim=ACTION_DIM, proprio_dim=0,
        img_dim=512, graph_dim=512, goal_dim=512, img_mod=[], graph_mod=list(graph_mod),
        use_graph_fusion=True, embed_dim=embed_dim, embed_pdrob=0.0, attn_pdrop=0.0,
        resid_pdrop=0.0, mlp_pdrop=0.0, n_dec_layers=2, n_enc_layers=2, n_heads=4,
        max_graph_nodes=max_graph_nodes,
    ).eval()


def _inputs():
    return (torch.randn(B, ACT_WINDOW, ACTION_DIM), {"lang": torch.randn(B, 512)},
            torch.rand(B) + 0.1)


def test_pooled_readout_forward():
    """A pooled graph encoder emits one [B, 1, D] token with an all-valid mask."""
    m = _build()
    act, goal, sigma = _inputs()
    state = {"obs_graph": {MOD: (torch.randn(B, 1, 512), torch.ones(B, 1, dtype=torch.bool))}}
    with torch.no_grad():
        out = m(state, act, goal, sigma)
    assert out.shape == (B, ACT_WINDOW, ACTION_DIM), out.shape


def test_per_node_readout_forward():
    m = _build()
    act, goal, sigma = _inputs()
    state = {"obs_graph": {MOD: (torch.randn(B, 7, 512), torch.ones(B, 7, dtype=torch.bool))}}
    with torch.no_grad():
        out = m(state, act, goal, sigma)
    assert out.shape == (B, ACT_WINDOW, ACTION_DIM), out.shape


def test_padded_graph_nodes_do_not_affect_output():
    """Tokens masked out must be invisible; unmasked ones must matter."""
    m = _build()
    act, goal, sigma = _inputs()
    tokens = torch.randn(B, 7, 512)
    mask = torch.ones(B, 7, dtype=torch.bool)
    mask[0, 4:] = False

    with torch.no_grad():
        base = m({"obs_graph": {MOD: (tokens, mask)}}, act, goal, sigma)

        padded = tokens.clone()
        padded[0, 5] += 100.0                     # a masked slot
        out_padded = m({"obs_graph": {MOD: (padded, mask)}}, act, goal, sigma)

        real = tokens.clone()
        real[0, 2] += 100.0                       # a real node
        out_real = m({"obs_graph": {MOD: (real, mask)}}, act, goal, sigma)

    assert torch.allclose(base, out_padded, atol=1e-6), "masked token leaked into the output"
    assert not torch.allclose(base, out_real, atol=1e-6), "real token had no effect"


def test_seq_length_covers_max_graph_nodes():
    m = _build(max_graph_nodes=32)
    assert m.model.get_seq_length(1) == 1 + 32
    assert m.model.pos_emb.shape[1] == 1 + 32
    act, goal, sigma = _inputs()
    state = {"obs_graph": {MOD: (torch.randn(B, 32, 512), torch.ones(B, 32, dtype=torch.bool))}}
    with torch.no_grad():
        out = m(state, act, goal, sigma)
    assert out.shape == (B, ACT_WINDOW, ACTION_DIM)


def test_pooled_mode_keeps_legacy_sequence_length():
    """With max_graph_nodes == obs_window the token budget matches the pre-change model,
    so checkpoints trained before the per-node readout still load."""
    m = _build(max_graph_nodes=1, embed_dim=512)
    assert m.model.get_seq_length(1) == 2
    assert tuple(m.model.pos_emb.shape) == (1, 2, 512)


def test_multiple_graph_modalities():
    m = _build(graph_mod=("a", "b"), max_graph_nodes=16)
    act, goal, sigma = _inputs()
    state = {"obs_graph": {
        "a": (torch.randn(B, 16, 512), torch.ones(B, 16, dtype=torch.bool)),
        "b": (torch.randn(B, 1, 512), torch.ones(B, 1, dtype=torch.bool)),
    }}
    with torch.no_grad():
        out = m(state, act, goal, sigma)
    assert m.model.get_seq_length(1) == 1 + 16 + 16
    assert out.shape == (B, ACT_WINDOW, ACTION_DIM)


def test_cross_attention_is_not_causal():
    """Every action query must see the whole state context, not a prefix of it."""
    torch.manual_seed(0)
    D, L_q, L_k = 32, 10, 5
    dec = TransformerFiLMDecoder(embed_dim=D, n_heads=4, attn_pdrop=0.0, resid_pdrop=0.0,
                                 n_layers=2, film_cond_dim=D, mlp_pdrop=0.0,
                                 use_cross_attention=True).eval()
    assert dec.blocks[0].cross_att.causal is False, "cross attention must not be causal"
    assert dec.blocks[0].attn.causal is True, "self attention should stay causal"

    x, c, ctx = torch.randn(2, L_q, D), torch.randn(2, 1, D), torch.randn(2, L_k, D)
    with torch.no_grad():
        base = dec(x, c, ctx)
        ctx2 = ctx.clone(); ctx2[:, -1] += 5.0     # perturb the LAST context token
        pert = dec(x, c, ctx2)
    delta = (base - pert).abs().amax(dim=-1)[0]
    assert bool(delta[0] > 1e-6), "action query 0 could not see the last context token"


def test_cross_attn_mask_is_independent_of_self_attn_mask():
    torch.manual_seed(0)
    D, L_q, L_k = 32, 10, 5
    dec = TransformerFiLMDecoder(embed_dim=D, n_heads=4, attn_pdrop=0.0, resid_pdrop=0.0,
                                 n_layers=2, film_cond_dim=D, mlp_pdrop=0.0,
                                 use_cross_attention=True).eval()
    x, c, ctx = torch.randn(2, L_q, D), torch.randn(2, 1, D), torch.randn(2, L_k, D)
    m = torch.ones(2, 1, 1, L_k, dtype=torch.bool); m[:, :, :, -1] = False
    with torch.no_grad():
        ctx2 = ctx.clone(); ctx2[:, -1] += 5.0
        assert torch.allclose(dec(x, c, ctx, cross_attn_mask=m),
                              dec(x, c, ctx2, cross_attn_mask=m), atol=1e-6), \
            "masking the perturbed key should make the perturbation irrelevant"


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
