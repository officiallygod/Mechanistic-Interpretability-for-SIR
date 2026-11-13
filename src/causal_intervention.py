"""
Causal Intervention & Evaluation Module
Implements feature clamping and ablation for testing Discovery, Removal,
and Consequence conditions on robot policies during closed-loop rollouts.
"""

from typing import List, Optional, Set
import torch
import torch.nn as nn
from src.sae import SparseAutoencoder


class CausalInterventionHook:
    """
    Hooks into policy activations during rollout to replace raw activations with
    SAE-reconstructed activations where specific object features are ablated.
    """

    def __init__(
        self,
        target_layer: nn.Module,
        sae: SparseAutoencoder,
        ablated_features: Optional[Set[int]] = None,
    ):
        self.target_layer = target_layer
        self.sae = sae
        self.ablated_features = ablated_features or set()
        self.hook_handle = None
        self.active = True
        self._register()

    def set_ablated_features(self, feature_indices: Set[int]):
        """Sets which SAE features should be clamped to zero."""
        self.ablated_features = set(feature_indices)

    def _hook_fn(self, module: nn.Module, input_tensor, output_tensor: torch.Tensor):
        if not self.active or not self.ablated_features:
            return output_tensor

        # output_tensor shape: [B, C, H, W] = [B, 512, 7, 7]
        b, c, h, w = output_tensor.shape
        # Reshape to [B * H * W, 512] for SAE
        x = output_tensor.permute(0, 2, 3, 1).reshape(-1, c)

        with torch.no_grad():
            f = self.sae.encode(x)
            # Causal ablation: zero out specified feature indices
            for feat_idx in self.ablated_features:
                if feat_idx < f.shape[-1]:
                    f[:, feat_idx] = 0.0
            x_reconstructed = self.sae.decode(f)

        # Restore original shape [B, C, H, W]
        x_intervened = x_reconstructed.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x_intervened

    def _register(self):
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
