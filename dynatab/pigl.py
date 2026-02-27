# This part of code implements Positional Importance Gating Layer (PIGL)
from __future__ import annotations

import torch
import torch.nn as nn


class PositionalImportanceGatingLayer(nn.Module):
    """
    PIGL: Positional Importance Gating Layer.

    Expected usage:
        pigl = PositionalImportanceGatingLayer(num_features=m, embedding_dim=d).to(device)
        x_pigl = pigl(x_ope, importance_scores)  # [B, m, d]

    Inputs:
      - x_ope: [B, m, d]
      - importance_scores: [m] (float)  (e.g., variance per feature)

    Output:
      - x_pigl: [B, m, d]
    """

    def __init__(self, num_features: int, embedding_dim: int):
        super().__init__()
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)

        # Learnable per-feature gate parameters
        self.gate_weights = nn.Parameter(torch.randn(self.num_features))   # [m]
        self.gate_biases = nn.Parameter(torch.zeros(self.num_features))    # [m]

    def forward(self, x_ope: torch.Tensor, importance_scores: torch.Tensor) -> torch.Tensor:
        if x_ope.dim() != 3:
            raise ValueError(f"x_ope must be [B,m,d]. Got {tuple(x_ope.shape)}")
        B, m, d = x_ope.shape
        if m != self.num_features:
            raise ValueError(f"num_features mismatch: expected {self.num_features}, got {m}")
        if d != self.embedding_dim:
            raise ValueError(f"embedding_dim mismatch: expected {self.embedding_dim}, got {d}")

        if importance_scores.dim() != 1 or importance_scores.numel() != m:
            raise ValueError(f"importance_scores must be [m] with m={m}. Got {tuple(importance_scores.shape)}")

        gamma = importance_scores.to(device=x_ope.device, dtype=x_ope.dtype)  # [m]
        gates = torch.sigmoid(self.gate_weights.to(x_ope.dtype) * gamma + self.gate_biases.to(x_ope.dtype))  # [m]

        gates = gates.unsqueeze(0).unsqueeze(-1)  # [1, m, 1]
        return x_ope * gates  # [B, m, d]

# ---- Aliases for compatibility (do NOT change the class) ----
PIGL = PositionalImportanceGatingLayer
__all__ = ["PositionalImportanceGatingLayer", "PIGL"]