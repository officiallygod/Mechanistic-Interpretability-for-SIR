"""
Activation Recorder Module
Hooks into the ResNet-18 vision backbone before global pooling to record
the 7x7 spatial feature map [Batch, 512, 7, 7] during policy rollouts.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class ActivationRecorder:
    """
    Registers a PyTorch forward hook to capture and cache intermediate activations
    from a target layer (e.g. ResNet-18 layer4).
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "vision_model.layer4"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations: List[torch.Tensor] = []
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_hook()

    def _hook_fn(self, module: nn.Module, input_tensor, output_tensor: torch.Tensor):
        # Detach and store on CPU to avoid GPU VRAM exhaustion during long rollouts
        self.activations.append(output_tensor.detach().cpu())

    def _register_hook(self):
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_module = module
                break

        if target_module is None:
            # Fallback search for layer4 if prefixed differently
            for name, module in self.model.named_modules():
                if name.endswith("layer4"):
                    target_module = module
                    self.target_layer_name = name
                    break

        if target_module is None:
            raise ValueError(f"Could not find target layer '{self.target_layer_name}' in model.")

        self.hook_handle = target_module.register_forward_hook(self._hook_fn)

    def clear(self):
        """Clears the recorded activation cache."""
        self.activations.clear()

    def get_activations(self) -> torch.Tensor:
        """
        Returns stacked tensor of all recorded activations.
        Shape: [Total_Timesteps, Channels (512), H (7), W (7)]
        """
        if not self.activations:
            return torch.empty(0)
        return torch.cat(self.activations, dim=0)

    def remove(self):
        """Removes the forward hook from the PyTorch model."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
