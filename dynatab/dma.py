# dma.py
# This part of the code implements Dynamic Masked Attention (DMA)

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def create_dma_mask(global_order: torch.Tensor) -> torch.Tensor:
    """
    Create a causal DMA mask [m, m] for the *current* feature-token order.

    IMPORTANT:
      - DFO returns a permutation and you reorder X accordingly.
      - After reordering, the sequence index (0..m-1) is the notion of "time/order".
      - Therefore the correct causal mask is lower-triangular in sequence positions,
        NOT based on comparing the permutation values.

    Returns:
      attn_mask[i, j] = 0.0   if j <= i   (key j allowed for query i)
                     = -inf  otherwise
    """
    if global_order.dim() != 1:
        raise ValueError(f"global_order must be [m]. Got {tuple(global_order.shape)}")

    m = int(global_order.numel())
    device = global_order.device

    i = torch.arange(m, device=device)
    j = torch.arange(m, device=device)
    allowed = (i.unsqueeze(1) >= j.unsqueeze(0))  # lower-triangular [m,m]

    mask = torch.full((m, m), float("-inf"), device=device, dtype=torch.float32)
    mask[allowed] = 0.0
    return mask


class DynamicMaskedAttention(nn.Module):
    """
    DMA: Multi-Head Self-Attention with a precomputed (or internally computed) DMA mask.

    Recommended usage (centralize global_order handling outside):
        dma_mask = create_dma_mask(global_order)        # [m,m]
        dma = DynamicMaskedAttention(embed_dim=d, num_heads=h).to(device)
        x_dma = dma(x_pigl, attn_mask=dma_mask)         # [B,m,d]

    Inputs:
      - x: [B, m, d]
      - attn_mask: [m, m] float additive mask (0.0 / -inf). If None, you must pass global_order.
      - global_order: [m] (optional) used only when attn_mask is None.
      - key_padding_mask: [B, m] bool mask (True = pad) (optional)

    Output:
      - x_out: [B, m, d]
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            bias=bias,
            batch_first=True,  # input/output: [B, m, d]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        global_order: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,m,d]. Got {tuple(x.shape)}")
        B, m, d = x.shape
        if d != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: expected {self.embed_dim}, got {d}")

        # Build / validate attention mask
        if attn_mask is None:
            if global_order is None:
                raise ValueError("Provide either attn_mask or global_order.")
            if global_order.dim() != 1:
                raise ValueError(f"global_order must be [m]. Got {tuple(global_order.shape)}")
            if global_order.numel() != m:
                raise ValueError(f"global_order length mismatch: expected m={m}, got {global_order.numel()}")
            attn_mask = create_dma_mask(global_order.to(device=x.device))
        else:
            if attn_mask.shape != (m, m):
                raise ValueError(f"attn_mask must be [m,m] = {(m, m)}. Got {tuple(attn_mask.shape)}")
            attn_mask = attn_mask.to(device=x.device, dtype=torch.float32)

        # key_padding_mask: [B,m] bool (True means ignore)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, m):
                raise ValueError(
                    f"key_padding_mask must be [B,m] = {(B, m)}. Got {tuple(key_padding_mask.shape)}"
                )
            key_padding_mask = key_padding_mask.to(device=x.device)

        out, weights = self.mha(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return (out, weights) if need_weights else out