"""
Grounding Module
Maps SAE feature activation maps across the 7x7 spatial grid to camera pixel space
and computes spatial alignment (IoU / mask overlap) with ground-truth simulator bounding boxes.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def project_grid_to_bbox_iou(
    feature_heatmap_7x7: np.ndarray,
    target_bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int] = (224, 224),
    threshold_percentile: float = 80.0,
) -> float:
    """
    Computes Intersection-over-Union between an upsampled feature activation heatmap
    and a simulator 2D bounding box [xmin, ymin, xmax, ymax].
    """
    h_img, w_img = image_shape
    # Upsample 7x7 heatmap to full image resolution
    tensor_map = torch.from_numpy(feature_heatmap_7x7).unsqueeze(0).unsqueeze(0).float()
    upsampled = F.interpolate(tensor_map, size=(h_img, w_img), mode="bilinear", align_corners=False)
    heatmap_full = upsampled.squeeze().numpy()

    # Create binary mask of top active regions
    thresh = np.percentile(heatmap_full, threshold_percentile)
    pred_mask = heatmap_full >= thresh

    # Create target bounding box mask
    xmin, ymin, xmax, ymax = target_bbox
    target_mask = np.zeros((h_img, w_img), dtype=bool)
    target_mask[max(0, ymin):min(h_img, ymax), max(0, xmin):min(w_img, xmax)] = True

    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_latent_kept_set(
    active_feature_indices: List[int],
    feature_to_object_map: Dict[int, str],
) -> List[str]:
    """
    Given the list of active SAE features at timestep t and a precomputed dictionary
    mapping features to ground-truth object labels, returns the Latent Kept Set K_latent.
    """
    kept_objects = set()
    for feat_idx in active_feature_indices:
        if feat_idx in feature_to_object_map:
            kept_objects.add(feature_to_object_map[feat_idx])
    return sorted(list(kept_objects))
