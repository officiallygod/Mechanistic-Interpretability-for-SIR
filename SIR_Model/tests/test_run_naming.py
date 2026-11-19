"""Tests for the wandb run naming derived from a resolved config.

This existed as a bug for a long time: main.py read `cfg.run_name` while the key is
`cfg.wandb.run_name`, so every `wandb.run_name=` on the command line was silently
ignored and every run got a random codename. Nothing failed, the names were just
useless — which is why it went unnoticed across a dozen runs. Hence tests.

Run: python tests/test_run_naming.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.run_naming import dataset_tag, derive_run_name, variant_name


class Skip(Exception):
    pass


def _cfg(encoder_target="networks.graph_encoder.gnn.Multi_XAI_GNN",
         strategy="topk_PerTask", cbm="nodes_att", graph=("bb_coordinates",),
         image=(), seed=43, split=None, exclude_scene_d=False):
    return {
        "trainer": {"seed": seed},
        "manager": {"graph_modalities": list(graph), "image_modalities": list(image),
                    "split": split, "exclude_scene_d": exclude_scene_d},
        "method": {"graph_encoder": {
            "_target_": encoder_target,
            "use_cbm_loss": cbm,
            "sparsification_layer": {"sampling_strategy": strategy},
        }},
    }


def test_the_four_variants_we_actually_run():
    assert derive_run_name(_cfg()) == "SIR_s43"
    assert derive_run_name(_cfg(strategy="topk", cbm=None)) == "TopK_s43"
    assert derive_run_name(_cfg(encoder_target="networks.graph_encoder.gnn.Multi_GNN")) == "GNN_s43"
    assert derive_run_name(_cfg(graph=(), image=("static",))) == "Image_s43"


def test_seed_is_carried_through():
    for seed in (43, 44, 101010):
        assert derive_run_name(_cfg(seed=seed)) == f"SIR_s{seed}"


def test_per_task_budget_without_cbm_loss_is_not_called_sir():
    """`topk_PerTask` alone is half the method — the CBM loss is the other half.
    Labelling it SIR would silently mix an ablation into the headline numbers."""
    assert variant_name(_cfg(cbm=None)) == "PerTask"
    assert variant_name(_cfg(cbm="nodes_att")) == "SIR"


def test_threshold_variant():
    assert derive_run_name(_cfg(strategy="threshold", cbm=None)) == "Threshold_s43"


def test_vgib_is_named_for_being_unsupervised():
    """VGIB's whole claim is that it selects without the relevance table, so a run that
    quietly left the CBM loss on must not share a name with one that did not — the two
    answer opposite questions."""
    assert derive_run_name(_cfg(strategy="vgib", cbm=None)) == "VGIB_s43"
    assert derive_run_name(_cfg(strategy="vgib", cbm="nodes_att")) == "VGIB+CBM_s43"


def test_graph_plus_image_is_not_mistaken_for_the_graph_baseline():
    assert derive_run_name(_cfg(image=("static",))) == "SIR+Image_s43"


def test_unknown_encoder_falls_back_to_its_class_name():
    cfg = _cfg(encoder_target="networks.graph_encoder.gnn.Some_New_Encoder", strategy="wat")
    assert derive_run_name(cfg) == "Some_New_Encoder_s43"


def test_missing_seed_degrades_to_the_bare_variant():
    cfg = _cfg()
    cfg["trainer"] = {}
    assert derive_run_name(cfg) == "SIR"


def test_d_d_names_are_unchanged_by_the_split_tag():
    """The D->D results were produced before the tag existed; adding one now would
    orphan those names in wandb."""
    assert derive_run_name(_cfg(split="D_D")) == "SIR_s43"
    assert derive_run_name(_cfg(split=None)) == "SIR_s43"


def test_abc_d_gets_its_own_namespace():
    assert derive_run_name(_cfg(split="ABC_D")) == "SIR_ABC_D_s43"
    assert derive_run_name(_cfg(split="ABC_D", strategy="topk", cbm=None)) == "TopK_ABC_D_s43"
    assert derive_run_name(_cfg(split="ABC_D", graph=(), image=("static",))) == "Image_ABC_D_s43"


def test_excluding_scene_d_is_a_different_run_than_keeping_it():
    """Numbers from the two are not comparable, so they must not collide."""
    assert derive_run_name(_cfg(split="ABC_D", exclude_scene_d=True)) == "SIR_ABC_Dx_s43"
    assert dataset_tag(_cfg(split="ABC_D", exclude_scene_d=True)) != \
        dataset_tag(_cfg(split="ABC_D", exclude_scene_d=False))


def test_robocasa_configs_have_no_split_and_are_untouched():
    cfg = _cfg()
    cfg["manager"] = {"graph_modalities": ["bb_coordinates"], "image_modalities": []}
    assert dataset_tag(cfg) is None
    assert derive_run_name(cfg) == "SIR_s43"


if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    failed = []
    skipped = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except Skip as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed) - skipped}/{len(tests)} passed, {skipped} skipped")
    sys.exit(0 if not failed else 1)
