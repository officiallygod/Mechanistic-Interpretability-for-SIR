"""
Sparse Autoencoder (SAE) Module
Decomposes dense activation vectors into overcomplete sparse feature representations.
Supports Top-K and L1 sparsity penalties for monosemantic dictionary learning.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Standard / Top-K Sparse Autoencoder for mechanistic interpretability.
    Maps activation vector x in R^d -> sparse feature vector f in R^M (M >> d) -> x_hat in R^d.
    """

    def __init__(
        self,
        d_in: int = 512,
        d_sae: int = 4096,
        sparsity_type: str = "topk",
        k: int = 32,
        l1_coeff: float = 1e-3,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.sparsity_type = sparsity_type.lower()
        self.k = k
        self.l1_coeff = l1_coeff

        # Geometric initialization (tied/untied weights)
        self.w_enc = nn.Linear(d_in, d_sae, bias=True)
        self.w_dec = nn.Linear(d_sae, d_in, bias=True)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Initialize decoder weights with unit norm columns
        with torch.no_grad():
            self.w_dec.weight.data = F.normalize(self.w_dec.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes dense activation x into sparse feature vector f.
        Input x: [..., d_in]
        Output f: [..., d_sae]
        """
        # Center with respect to decoder bias
        x_centered = x - self.b_dec
        pre_acts = self.w_enc(x_centered)

        if self.sparsity_type == "topk":
            # Retain only top-k highest activations, zero out the rest
            val, idx = torch.topk(pre_acts, self.k, dim=-1)
            f = torch.zeros_like(pre_acts)
            f.scatter_(-1, idx, F.relu(val))
            return f
        else:
            # Standard ReLU with L1 penalty during loss computation
            return F.relu(pre_acts)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs activation vector from sparse feature vector f.
        Input f: [..., d_sae]
        Output x_hat: [..., d_in]
        """
        return self.w_dec(f) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            x_hat: Reconstructed activation tensor
            f: Sparse feature activations
            loss: Combined reconstruction loss + sparsity loss
        """
        f = self.encode(x)
        x_hat = self.decode(f)

        # Mean squared error reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        if self.sparsity_type == "l1":
            l1_loss = self.l1_coeff * f.abs().sum(dim=-1).mean()
            total_loss = recon_loss + l1_loss
        else:
            total_loss = recon_loss

        return x_hat, f, total_loss
