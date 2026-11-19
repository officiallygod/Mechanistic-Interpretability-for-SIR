"""Derive a readable wandb run name from the resolved config.

Every run in this project is "some variant at some seed", but that identity is spread
across four config keys — the graph encoder's target class, its sampling strategy,
whether the CBM loss is on, and which modalities are non-empty. Reconstructing it by eye
from a wandb config panel is tedious, and the alternative (typing `wandb.run_name=` on
every command) is exactly the sort of thing that gets forgotten or mistyped when a
sweep is launched at 2am.

So the name is computed from the config that actually ran, which means it cannot
disagree with it.
"""

_ENCODER_CLASS_KEY = "_target_"


def _sampling_strategy(graph_encoder):
    layer = graph_encoder.get("sparsification_layer") or {}
    return layer.get("sampling_strategy")


def variant_name(cfg):
    """Short label for the method variant, e.g. "SIR", "TopK", "GNN", "Image"."""
    manager = cfg.get("manager") or {}
    uses_graph = bool(manager.get("graph_modalities"))
    uses_image = bool(manager.get("image_modalities"))

    if not uses_graph:
        # No graph at all: the only signal left is the camera stream.
        return "Image" if uses_image else "Prop"

    encoder = (cfg.get("method") or {}).get("graph_encoder") or {}
    target = str(encoder.get(_ENCODER_CLASS_KEY, "")).rsplit(".", 1)[-1]

    if target == "Multi_GNN":
        name = "GNN"
    else:
        strategy = _sampling_strategy(encoder)
        if strategy == "vgib":
            # VGIB is defined by *not* using the CBM loss — a run with both is neither
            # the unsupervised variant nor SIR, so name it so rather than letting it
            # pass for the thing it is meant to be compared against.
            name = "VGIB+CBM" if encoder.get("use_cbm_loss") else "VGIB"
        elif strategy == "topk_PerTask":
            # The per-task budget plus the CBM loss is the actual SIR method; the same
            # budget without the loss is only half of it, so do not call it SIR.
            name = "SIR" if encoder.get("use_cbm_loss") else "PerTask"
        elif strategy == "topk":
            name = "TopK"
        elif strategy == "threshold":
            name = "Threshold"
        else:
            name = target or "Graph"

    # A run with both graphs and images is neither baseline, so say so rather than
    # letting it masquerade as the graph-only variant.
    if uses_image:
        name = f"{name}+Image"

    return name


def dataset_tag(cfg):
    """Short label for the data a run was trained on, or None when there is nothing to say.

    Only CALVIN has more than one of these, and only its non-default split needs
    disambiguating: the D->D results were produced before this existed, so keeping D_D
    silent leaves those names untouched while ABC->D gets its own namespace. Without it
    an ABC->D SIR run and the D->D one both land on "SIR_s43" in wandb and the transfer
    result overwrites the in-distribution one.
    """
    manager = cfg.get("manager") or {}
    split = manager.get("split")
    if not split or split == "D_D":
        return None
    # A run that dropped the appended scene-D episodes is not comparable to one that
    # kept them, so the two must not share a name either.
    return f"{split}x" if manager.get("exclude_scene_d") else split


def derive_run_name(cfg):
    """`<Variant>[_<dataset>]_s<seed>`, e.g. "TopK_s43" or "SIR_ABC_Dx_s43"."""
    seed = (cfg.get("trainer") or {}).get("seed")
    parts = [variant_name(cfg)]

    tag = dataset_tag(cfg)
    if tag:
        parts.append(tag)
    if seed is not None:
        parts.append(f"s{seed}")

    return "_".join(parts)
