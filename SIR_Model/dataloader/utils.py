import torch

from utils.point_geom import (POINT_GEOM_CENTROID_OFFSET, POINT_GEOM_DIM,
                              POINT_GEOM_VISIBILITY_OFFSET)

GRAPH_MODALITY_LIST = ["one_hot_labels", "bb_coordinates", "cropped_image_feature",
                       "siglip_text_emb", "point_geom"]

# Per-modality width of one view's slice, and how the two views are combined.
#   "concat"    -> both views are kept side by side, so the fused width is 2x
#   "left_only" -> the feature is view invariant, so one copy is enough
# `cropped_image_feature` has no fixed width (it depends on the pretrained encoder)
# and its rule depends on whether the encoder already fused both views, so it is
# resolved at call time; None marks "resolve dynamically".
GRAPH_MODALITY_SPEC = {
    "one_hot_labels":       {"width": 37,   "fusion": "left_only"},
    "bb_coordinates":       {"width": 10,   "fusion": "concat"},
    "cropped_image_feature": {"width": None, "fusion": None},
    "siglip_text_emb":      {"width": 768,  "fusion": "left_only"},
    # Reconstructed from both views jointly and expressed in the robot-base frame, so
    # the two view files hold the same value for a node and one copy is enough.
    "point_geom":           {"width": POINT_GEOM_DIM, "fusion": "left_only"},
}

def combine_graph_modalities(graph_data, graph_mod, idx=None, j=None):
    if idx is None and j is None:
        base_graph = graph_data[graph_mod[0]].clone()
        feats = [graph_data[mod].x for mod in graph_mod]
    else:
        base_graph = graph_data[graph_mod[0]][idx][j].clone()
        feats = [graph_data[mod][idx][j].x for mod in graph_mod]
    base_graph.x = torch.cat(feats, dim=-1)
    
    return base_graph

def build_edge_weights(fused_x, offsets, src, dst):
    """Per-edge weight for a fully connected graph.

    Without 3D geometry every edge weighs 1, which gives the GNN no spatial structure at
    all. When `point_geom` is present the weight becomes 1 / (1 + ||Δcentroid||), so
    nearby objects attend to each other more strongly than distant ones.
    WeightedGATv2Conv already multiplies attention coefficients by this, and the
    sparsification module multiplies its own node gates on top, so the two compose.

    Edges touching a node whose geometry was never observed fall back to 1, because a
    zeroed centroid would otherwise fabricate a distance.
    """
    device = fused_x.device
    weights = torch.ones(src.shape[0], device=device, dtype=torch.float32)
    if "point_geom" not in offsets:
        return weights

    start, _ = offsets["point_geom"]
    centroids = fused_x[:, start + POINT_GEOM_CENTROID_OFFSET:
                           start + POINT_GEOM_CENTROID_OFFSET + 3]
    observed = fused_x[:, start + POINT_GEOM_VISIBILITY_OFFSET] > 0

    dist = torch.linalg.norm(centroids[dst] - centroids[src], dim=-1)
    valid = observed[src] & observed[dst]
    return torch.where(valid, 1.0 / (1.0 + dist), weights)


def fuse_graphs(graph_data_left, graph_data_right, mod, mod_order, step_idx=None, is_cropped_fusion=False):
    """Fuse the left/right view graphs for one modality key.

    `mod` is the dict key (a single modality name, or the joined name of several).
    `mod_order` is the ordered list of the modalities `mod` is built from — the same
    order `combine_graph_modalities` concatenated them in. It is passed explicitly
    rather than parsed back out of `mod`, because the joined name is ambiguous.
    """
    if step_idx is None:
        left_graph = graph_data_left[mod]
        right_graph = graph_data_right[mod]
    else:
        left_graph = graph_data_left[mod][step_idx]
        right_graph = graph_data_right[mod][step_idx]
    
    left_names = left_graph.node_names
    right_names = right_graph.node_names
    all_names = sorted(list(set(left_names) | set(right_names)))
    
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    num_nodes = len(all_names)
    
    feat_dim_l = left_graph.x.shape[1]
    feat_dim_r = right_graph.x.shape[1]
    
    # Create zero-filled tensors for the fused graph
    # Shape: [Total_Unique_Nodes, Left_Dim]
    x_l_mapped = torch.zeros((num_nodes, feat_dim_l), device=left_graph.x.device, dtype=left_graph.x.dtype)
    # Shape: [Total_Unique_Nodes, Right_Dim]
    x_r_mapped = torch.zeros((num_nodes, feat_dim_r), device=right_graph.x.device, dtype=right_graph.x.dtype)
    
    # Track which view saw which node, so a view-invariant feature can be read from
    # whichever view actually observed it rather than defaulting to an empty left row.
    present_l = torch.zeros(num_nodes, dtype=torch.bool, device=left_graph.x.device)
    present_r = torch.zeros(num_nodes, dtype=torch.bool, device=right_graph.x.device)

    # Map Left Features
    for src_i, name in enumerate(left_names):
        target_i = name_to_idx[name]
        x_l_mapped[target_i] = left_graph.x[src_i]
        present_l[target_i] = True

    # Map Right Features
    for src_i, name in enumerate(right_names):
        target_i = name_to_idx[name]
        x_r_mapped[target_i] = right_graph.x[src_i]
        present_r[target_i] = True

    fused_x, offsets = handle_fusing(x_l_mapped, x_r_mapped, mod_order, is_cropped_fusion,
                                     present_left=present_l, present_right=present_r)

    src, dst = torch.meshgrid(
        torch.arange(num_nodes, device=left_graph.x.device),
        torch.arange(num_nodes, device=left_graph.x.device),
        indexing="ij"
    )
    
    src = src.flatten()
    dst = dst.flatten()
    
    # Filter out self-loops (i != j)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    fused_edge_index = torch.stack((src, dst), dim=0)
    
    fused_edge_attr = build_edge_weights(fused_x, offsets, src, dst)

    fused_graph = type(left_graph)(
        x=fused_x,
        edge_index=fused_edge_index,
        edge_attr=fused_edge_attr,
        node_names=all_names  # Store the new unified list of names
    )
    
    return fused_graph

def handle_fusing(left, right, mod_order, cropped_fusion, present_left=None, present_right=None):
    """Combine the per-view feature blocks of every modality in `mod_order`.

    `left`/`right` are [N, F] with the modalities concatenated along the last axis in
    exactly `mod_order`, so slicing walks the list in order rather than trying to
    recover positions from a joined modality name. `cropped_image_feature` is the one
    modality whose width is not known statically (it depends on the pretrained
    encoder), so it absorbs whatever width the other modalities do not account for.

    `present_left`/`present_right` are optional [N] boolean masks marking which nodes
    each view actually saw. A view-invariant ("left_only") modality is normally read
    from the left view, but a node that only the right camera saw has an all-zero left
    row; the masks let it fall back to the right view instead of emitting zeros.

    Returns:
        fused:   [N, F_out]
        offsets: {modality: (start, width)} into `fused`, so callers can slice a
                 specific modality back out without recomputing the layout.
    """
    unknown = [m for m in mod_order if m not in GRAPH_MODALITY_SPEC]
    if unknown:
        raise ValueError(f"Unknown graph modality/modalities: {unknown}. "
                         f"Known: {list(GRAPH_MODALITY_SPEC)}")

    # `cropped_image_feature` has no static width — it takes up whatever is left over.
    dynamic = [m for m in mod_order if GRAPH_MODALITY_SPEC[m]["width"] is None]
    if len(dynamic) > 1:
        raise ValueError(f"At most one modality may have a dynamic width, got {dynamic}")
    known_len = sum(GRAPH_MODALITY_SPEC[m]["width"] for m in mod_order
                    if GRAPH_MODALITY_SPEC[m]["width"] is not None)
    dynamic_len = left.shape[-1] - known_len
    if dynamic and dynamic_len <= 0:
        raise ValueError(f"Feature width {left.shape[-1]} is too small for modalities "
                         f"{mod_order} (fixed-width part alone needs {known_len})")
    if not dynamic and dynamic_len != 0:
        raise ValueError(f"Feature width {left.shape[-1]} does not match modalities "
                         f"{mod_order} (expected {known_len})")

    fused_parts = []
    offsets = {}
    current_ptr = 0
    out_ptr = 0
    for m in mod_order:
        spec = GRAPH_MODALITY_SPEC[m]
        eff_len = dynamic_len if spec["width"] is None else spec["width"]

        l_slice = left[..., current_ptr: current_ptr + eff_len]
        r_slice = right[..., current_ptr: current_ptr + eff_len]

        fusion = spec["fusion"]
        if fusion is None:
            # cropped_image_feature: the 6-channel fusion encoder has already merged
            # both views into one feature, so only take it once in that case.
            fusion = "left_only" if cropped_fusion else "concat"

        if fusion == "concat":
            part = torch.cat((l_slice, r_slice), dim=-1)
        else:
            # View-invariant feature: both views hold the same value, so one copy is
            # enough — but prefer whichever view actually observed the node.
            if present_left is not None and present_right is not None:
                take_left = (present_left | ~present_right).unsqueeze(-1)
                part = torch.where(take_left, l_slice, r_slice)
            else:
                part = l_slice

        fused_parts.append(part)
        offsets[m] = (out_ptr, part.shape[-1])
        out_ptr += part.shape[-1]
        current_ptr += eff_len

    return torch.cat(fused_parts, dim=-1), offsets


def pad_sequence(sequence, target_length):
    """Helper to pad a sequence by repeating the last frame if it's too short."""
    current_len = sequence.shape[0]
    if current_len < target_length:
        pad_qty = target_length - current_len
        last_frame = sequence[-1].unsqueeze(0)
        padding = last_frame.repeat(pad_qty, *([1]*(len(sequence.shape)-1)))
        return torch.cat((sequence, padding), dim=0)
    return sequence
