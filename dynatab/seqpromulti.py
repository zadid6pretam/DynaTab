# seqpromulti.py
# Sequential Processor (Multi-class classification) for DynaTab.
# Backbones: DAE, LSTM, DAE-MHA-LSTM, Transformer, Mamba(SSM)
#
# PIPELINE DESIGN:
#   - DFO produces ONE global ordering Og (computed once per dataset/training setup).
#   - Og flows through OPE + PIGL upstream (independent of backbone choice).
#   - Attention backbones (Transformer, DAE-MHA-LSTM) additionally use Og to build DMA masks.
#   - Non-attention backbones (DAE/LSTM/Mamba) consume PIGL output directly.
#
# IMPORTANT IMPLEMENTATION FIX:
#   - Upstream may call processor(x_pigl) for non-attention backbones.
#   - Therefore forward() MUST allow global_ordering=None, and only REQUIRE it
#     for attention backbones.

from __future__ import annotations

from typing import Optional, Dict, Literal

import torch
import torch.nn as nn

from .dma import create_dma_mask

BackboneName = Literal["Transformer", "DAE", "LSTM", "DAE-MHA-LSTM", "Mamba"]


# -----------------------------
# Backbone selector (Multi-class)
# -----------------------------
class SequentialProcessorMulti(nn.Module):
    """
    Wrapper that standardizes the sequential processor interface.

    Forward signature is intentionally:
        forward(x_seq, global_ordering=None)

    because upstream code may call:
        - processor(x_pigl, global_ordering)  (attention backbones)
        - processor(x_pigl)                   (non-attention backbones)

    global_ordering is conceptually ALWAYS part of the pipeline (computed once),
    but only attention backbones NEED it inside this module to build the DMA mask.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        backbone: BackboneName = "Transformer",
        backbone_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone_name: BackboneName = backbone
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2 for multi-class classification.")

        backbone_kwargs = backbone_kwargs or {}

        if backbone == "Transformer":
            self.processor = TransformerModelMulti(
                input_dim=input_dim,
                num_classes=self.num_classes,
                **backbone_kwargs,
            )

        elif backbone == "DAE":
            self.processor = DAEOnlyMulti(
                d_model=input_dim,
                num_classes=self.num_classes,
                encoder_hidden=backbone_kwargs.get("encoder_hidden", 256),
                dropout_rate=backbone_kwargs.get("dropout_rate", 0.2),
            )

        elif backbone == "LSTM":
            self.processor = LSTMOnlyMulti(
                input_dim=input_dim,
                num_classes=self.num_classes,
                hidden_dim=backbone_kwargs.get("hidden_dim", 128),
                num_layers=backbone_kwargs.get("num_layers", 1),
                dropout_rate=backbone_kwargs.get("dropout_rate", 0.5),
            )

        elif backbone == "DAE-MHA-LSTM":
            self.processor = DAEMHALSTMMulti(
                input_dim=input_dim,
                num_classes=self.num_classes,
                encoder_hidden=backbone_kwargs.get("encoder_hidden", 256),
                proj_dim=backbone_kwargs.get("proj_dim", 128),
                mha_heads=backbone_kwargs.get("mha_heads", 4),
                lstm_hidden=backbone_kwargs.get("lstm_hidden", 256),
                dropout=backbone_kwargs.get("dropout", 0.2),
                window_size=backbone_kwargs.get("window_size", 32),
            )

        elif backbone == "Mamba":
            self.processor = MambaSSMTabularMulti(
                input_dim=input_dim,
                num_classes=self.num_classes,
                d_model=backbone_kwargs.get("d_model", 128),
                n_layers=backbone_kwargs.get("n_layers", 3),
                d_state=backbone_kwargs.get("d_state", 16),
                conv_kernel=backbone_kwargs.get("conv_kernel", 4),
                dropout=backbone_kwargs.get("dropout", 0.1),
            )

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(
        self,
        x_seq: torch.Tensor,
        global_ordering: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: [B, m, d] output of PIGL (always).
            global_ordering: [m] long, OPTIONAL at call site.
                REQUIRED only for attention backbones to build DMA masks.

        Returns:
            logits: [B, C] (no softmax)
        """
        if x_seq.dim() != 3:
            raise ValueError(f"x_seq must be [B,m,d]. Got {tuple(x_seq.shape)}")

        # Attention backbones require global ordering for DMA mask construction
        if self.backbone_name in ("Transformer", "DAE-MHA-LSTM"):
            if global_ordering is None:
                raise ValueError(
                    f"global_ordering is required for backbone={self.backbone_name} (DMA mask construction)."
                )
            if global_ordering.dim() != 1:
                raise ValueError(f"global_ordering must be [m]. Got {tuple(global_ordering.shape)}")
            if global_ordering.dtype != torch.long:
                global_ordering = global_ordering.long()

            return self.processor(x_seq, global_ordering)

        # Non-attention backbones ignore ordering at this stage.
        return self.processor(x_seq)


# -----------------------------
# Transformer (Multi-class)
# -----------------------------
class TransformerModelMulti(nn.Module):
    """
    Sliding-window + DMA-mask Transformer encoder for multi-class classification.

    Input:
      x: [B, m, input_dim]  (PIGL output)
      global_ordering: [m]  (REQUIRED)
    Output:
      logits: [B, C]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        window_size: int = 32,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.window_size = int(window_size)

        self.embed = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x: torch.Tensor, global_ordering: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,m,d]. Got {tuple(x.shape)}")
        if global_ordering is None:
            raise ValueError("global_ordering is required.")
        if global_ordering.dim() != 1:
            raise ValueError(f"global_ordering must be [m]. Got {tuple(global_ordering.shape)}")
        if global_ordering.dtype != torch.long:
            global_ordering = global_ordering.long()

        _, m, _ = x.shape
        x = self.embed(x)  # [B, m, d_model]

        full_mask = create_dma_mask(global_ordering.to(device=x.device)).to(dtype=torch.float32)  # [m,m]

        idx = torch.arange(m, device=x.device)
        band = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size  # [m,m]

        attn_mask = full_mask.masked_fill(~band, float("-inf"))  # [m,m]

        x = self.transformer(x, mask=attn_mask)  # [B, m, d_model]
        pooled = x.mean(dim=1)                   # [B, d_model]
        return self.head(pooled)                 # [B, C]


# -----------------------------
# DAE only (Multi-class)
# -----------------------------
class DAEOnlyMulti(nn.Module):
    """
    Pooled-feature autoencoder + multi-class head. Returns logits [B,C].
    """
    def __init__(self, d_model: int, num_classes: int, encoder_hidden: int = 256, dropout_rate: float = 0.2):
        super().__init__()
        self.num_classes = int(num_classes)

        self.encoder = nn.Sequential(
            nn.Linear(d_model, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.decoder = nn.Linear(encoder_hidden, d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,m,d]. Got {tuple(x.shape)}")
        x_flat = x.mean(dim=1)      # [B, d_model]
        h = self.encoder(x_flat)    # [B, hidden]
        recon = self.decoder(h)     # [B, d_model]
        return self.head(recon)     # [B, C]


# -----------------------------
# LSTM only (Multi-class)
# -----------------------------
class LSTMOnlyMulti(nn.Module):
    """
    LSTM over feature-sequence. Returns logits [B,C].
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,m,d]. Got {tuple(x.shape)}")
        out, _ = self.lstm(x)        # [B, m, hidden_dim]
        last = out[:, -1, :]         # [B, hidden_dim]
        return self.head(last)       # [B, C]


# -----------------------------
# DAE + DMA-masked MHA + LSTM (Multi-class)
# -----------------------------
class DAEMHALSTMMulti(nn.Module):
    """
    Denoising AE + local-window + DMA-mask MHA + LSTM (multi-class).
    Returns logits [B,C].

    Input:
      x: [B, m, input_dim]  (PIGL output)
      global_ordering: [m]  (REQUIRED)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        encoder_hidden: int = 256,
        proj_dim: int = 128,
        mha_heads: int = 4,
        lstm_hidden: int = 256,
        dropout: float = 0.2,
        window_size: int = 32,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.window_size = int(window_size)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Linear(encoder_hidden, input_dim)

        self.projection = nn.Linear(input_dim, proj_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=mha_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(proj_dim, lstm_hidden, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(lstm_hidden)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x: torch.Tensor, global_ordering: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,m,d]. Got {tuple(x.shape)}")
        if global_ordering is None:
            raise ValueError("global_ordering is required.")
        if global_ordering.dim() != 1:
            raise ValueError(f"global_ordering must be [m]. Got {tuple(global_ordering.shape)}")
        if global_ordering.dtype != torch.long:
            global_ordering = global_ordering.long()

        _, m, _ = x.shape

        enc = self.encoder(x)        # [B, m, encoder_hidden]
        dec = self.decoder(enc)      # [B, m, input_dim]
        proj = self.projection(dec)  # [B, m, proj_dim]

        full_mask = create_dma_mask(global_ordering.to(device=x.device)).to(dtype=torch.float32)  # [m,m]

        idx = torch.arange(m, device=x.device)
        band = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size  # [m,m]

        attn_mask = full_mask.masked_fill(~band, float("-inf"))  # [m,m]

        attn_out, _ = self.mha(proj, proj, proj, attn_mask=attn_mask)  # [B, m, proj_dim]
        lstm_out, _ = self.lstm(attn_out)                               # [B, m, lstm_hidden]
        last = lstm_out[:, -1, :]                                       # [B, lstm_hidden]

        normed = self.batch_norm(last)
        return self.head(normed)                                        # [B, C]


# -----------------------------
# True SSM-style Mamba (Multi-class)
# -----------------------------
class MambaSSMBlock(nn.Module):
    """
    Same SSM-style block as in seqprobinary.py, reused here.
    Input/Output: [B, m, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.dropout = nn.Dropout(dropout)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_model)

        self.dwconv = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_model,
            bias=True,
        )

        self.A_log = nn.Parameter(torch.randn(self.d_model, self.d_state) * 0.02)
        self.B = nn.Parameter(torch.randn(self.d_model, self.d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(self.d_model, self.d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(self.d_model))
        self.dt_log = nn.Parameter(torch.zeros(self.d_model))

        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        B, m, d = x.shape
        if d != self.d_model:
            raise ValueError(f"SSM scan d mismatch: expected {self.d_model}, got {d}")

        dt = torch.nn.functional.softplus(self.dt_log).to(x.dtype)
        A = -torch.exp(self.A_log).to(x.dtype)

        a_disc = torch.exp(A * dt.unsqueeze(-1))
        b_disc = self.B.to(x.dtype) * dt.unsqueeze(-1)

        C = self.C.to(x.dtype)
        D = self.D.to(x.dtype)

        state = torch.zeros(B, d, self.d_state, device=x.device, dtype=x.dtype)

        ys = []
        for t in range(m):
            xt = x[:, t, :]
            state = state * a_disc.unsqueeze(0) + (b_disc.unsqueeze(0) * xt.unsqueeze(-1))
            yt = (state * C.unsqueeze(0)).sum(dim=-1) + (D.unsqueeze(0) * xt)
            ys.append(yt)

        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MambaSSMBlock expects [B,m,d]. Got {tuple(x.shape)}")

        residual = x

        u_gate = self.in_proj(x)
        u, gate = u_gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        u_t = u.transpose(1, 2)
        u_t = self.dwconv(u_t)
        u_t = u_t[:, :, : u.shape[1]]
        u = u_t.transpose(1, 2)

        u = torch.nn.functional.silu(u)
        y = self._ssm_scan(u)

        y = gate * y
        y = self.out_proj(y)
        y = self.dropout(y)
        return self.norm(y + residual)


class MambaSSMTabularMulti(nn.Module):
    """
    SSM-style Mamba backbone for tabular feature-sequence.
    Consumes x: [B,m,input_dim] and returns logits [B,C].
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaSSMBlock(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(int(n_layers))
        ])

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MambaSSMTabularMulti expects [B,m,d]. Got {tuple(x.shape)}")

        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)

        x = x.mean(dim=1)
        return self.head(x)