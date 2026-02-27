# This part of the code implements Order-Aware Positional Embedding (OPE)
from __future__ import annotations

import torch
import torch.nn as nn


class OrderAwarePositionalEmbedding(nn.Module):
    """
    OPE: adds an order-aware positional embedding to reordered features.

    Expected usage:
        ope_layer = OrderAwarePositionalEmbedding(num_features=m, embedding_dim=d).to(device)
        x_ope = ope_layer(x_reordered, global_order)   # -> [B, m, d]

    Inputs:
      - x_reordered: [B, m, d] (preferred). If [B, m], it will be treated as [B, m, 1].
      - global_order: [m] or [B, m] (long)

    Output:
      - x_ope: [B, m, d]  (same d as embedding_dim)
    """

    def __init__(self, num_features: int, embedding_dim: int):
        super().__init__()
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.positional_embedding = nn.Embedding(self.num_features, self.embedding_dim)

    def forward(self, x_reordered: torch.Tensor, global_order: torch.Tensor) -> torch.Tensor:
        if x_reordered.dim() == 2:
            x_reordered = x_reordered.unsqueeze(-1)  # [B, m] -> [B, m, 1]

        if x_reordered.dim() != 3:
            raise ValueError(f"x_reordered must be [B,m,d]. Got {tuple(x_reordered.shape)}")

        B, m, d = x_reordered.shape
        if m != self.num_features:
            raise ValueError(f"num_features mismatch: expected {self.num_features}, got {m}")
        if d != self.embedding_dim:
            raise ValueError(f"embedding_dim mismatch: expected {self.embedding_dim}, got {d}")

        if global_order.dtype != torch.long:
            global_order = global_order.long()

        if global_order.dim() == 1:
            if global_order.numel() != m:
                raise ValueError(f"global_order length mismatch: expected {m}, got {global_order.numel()}")
            global_order = global_order.unsqueeze(0).expand(B, -1)  # [B, m]
        elif global_order.dim() == 2:
            if global_order.shape != (B, m):
                raise ValueError(f"global_order must be [B,m]. Got {tuple(global_order.shape)}")
        else:
            raise ValueError(f"global_order must be [m] or [B,m]. Got {tuple(global_order.shape)}")

        pe = self.positional_embedding(global_order)  # [B, m, d]
        return x_reordered + pe