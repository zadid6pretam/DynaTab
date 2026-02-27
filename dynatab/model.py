# dynatab/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Literal, Union

import torch
import torch.nn as nn

from .ope import OrderAwarePositionalEmbedding
from .pigl import PositionalImportanceGatingLayer as PIGL  # keep your class name
from .seqprobinary import SequentialProcessorBinary
from .seqpromulti import SequentialProcessorMulti
from .seqproregression import SequentialProcessorRegression

BackboneName = Literal["Transformer", "DAE", "LSTM", "DAE-MHA-LSTM", "Mamba"]
TaskType = Literal["binary", "multiclass", "regression"]


@dataclass
class DynaTabOutputs:
    y_hat: torch.Tensor
    x_in: Optional[torch.Tensor] = None
    x_ope: Optional[torch.Tensor] = None
    x_pigl: Optional[torch.Tensor] = None


class _DynaTabBase(nn.Module):
    """
    Base DynaTab wiring:
        x_reordered -> input_proj -> OPE(global_order) -> PIGL(importance) -> backbone/head

    KEY CONTRACTS:
      - global_ordering is pipeline-level and MUST be provided every forward (independent of backbone).
      - x_reordered is assumed ALREADY in the same feature order as global_ordering.
      - This module NEVER recomputes DFO; DFO is an offline preprocessing step.

    Implementation:
      - Scalar tokens: [B,m] or [B,m,1] -> project 1 -> embedding_dim
      - Embedded tokens: [B,m,d] accepted if d == embedding_dim
      - SequentialProcessor* will internally require global_ordering only for attention backbones
        BUT we always pass it (safe) because forward() accepts optional and will ignore if not needed.
    """

    def __init__(
        self,
        task: TaskType,
        num_features: int,
        embedding_dim: int,
        backbone: BackboneName = "Transformer",
        backbone_kwargs: Optional[Dict] = None,
        num_classes: Optional[int] = None,  # only for multiclass
    ):
        super().__init__()
        self.task: TaskType = task
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.backbone: BackboneName = backbone
        self.backbone_kwargs = backbone_kwargs or {}

        # Project scalar feature value -> embedding space
        self.input_proj = nn.Linear(1, self.embedding_dim)

        # OPE + PIGL operate in embedding_dim space
        self.ope = OrderAwarePositionalEmbedding(num_features=self.num_features, embedding_dim=self.embedding_dim)
        self.pigl = PIGL(num_features=self.num_features, embedding_dim=self.embedding_dim)

        # Task-specific sequential processor
        if task == "binary":
            self.processor = SequentialProcessorBinary(
                input_dim=self.embedding_dim,
                backbone=backbone,
                backbone_kwargs=self.backbone_kwargs,
            )
        elif task == "multiclass":
            if num_classes is None:
                raise ValueError("num_classes is required for multiclass DynaTab.")
            self.processor = SequentialProcessorMulti(
                input_dim=self.embedding_dim,
                num_classes=int(num_classes),
                backbone=backbone,
                backbone_kwargs=self.backbone_kwargs,
            )
        elif task == "regression":
            self.processor = SequentialProcessorRegression(
                input_dim=self.embedding_dim,
                backbone=backbone,
                backbone_kwargs=self.backbone_kwargs,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    @staticmethod
    def _ensure_Bm_or_Bmd(x: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - [B,m]          -> [B,m,1]
          - [B,m,1]        -> [B,m,1]
          - [B,m,d]        -> [B,m,d]
        """
        if x.dim() == 2:
            return x.unsqueeze(-1)
        if x.dim() == 3:
            return x
        raise ValueError(f"x must be [B,m] or [B,m,d]. Got {tuple(x.shape)}")

    @staticmethod
    def _ensure_ordering(global_ordering: torch.Tensor, m: int, device: torch.device) -> torch.Tensor:
        if global_ordering is None:
            raise ValueError("global_ordering is required for DynaTab (pipeline-level).")
        if global_ordering.dim() != 1 or global_ordering.numel() != m:
            raise ValueError(f"global_ordering must be [m] with m={m}. Got {tuple(global_ordering.shape)}")
        if global_ordering.dtype != torch.long:
            global_ordering = global_ordering.long()
        return global_ordering.to(device=device)

    @staticmethod
    def _ensure_importance(importance_scores: torch.Tensor, m: int, device: torch.device) -> torch.Tensor:
        if importance_scores is None:
            raise ValueError("importance_scores is required.")
        if importance_scores.dim() != 1 or importance_scores.numel() != m:
            raise ValueError(f"importance_scores must be [m] with m={m}. Got {tuple(importance_scores.shape)}")
        return importance_scores.to(device=device, dtype=torch.float32)

    def forward(
        self,
        x_reordered: torch.Tensor,
        global_ordering: torch.Tensor,
        importance_scores: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, DynaTabOutputs]:
        """
        Args:
            x_reordered: [B,m] or [B,m,1] or [B,m,d] (already reordered)
            global_ordering: [m] long (REQUIRED, pipeline-level)
            importance_scores: [m] float (e.g., variance on TRAIN fold)
            return_intermediates: if True, return DynaTabOutputs

        Returns:
            binary:     logits [B,1]
            multiclass: logits [B,C]
            regression: [B,out_dim]
        """
        # ---- 0) basic shape + device ----
        x = self._ensure_Bm_or_Bmd(x_reordered)
        dev = x.device
        m = self.num_features

        # ---- 1) ordering + importance normalize ----
        global_ordering = self._ensure_ordering(global_ordering, m=m, device=dev)
        importance_scores = self._ensure_importance(importance_scores, m=m, device=dev)

        # ---- 2) project if needed ----
        # Scalar input: [B,m,1] -> [B,m,embedding_dim]
        if x.size(-1) == 1:
            x_in = self.input_proj(x)
        else:
            if x.size(-1) != self.embedding_dim:
                raise ValueError(
                    f"Input last-dim must be 1 or embedding_dim={self.embedding_dim}. Got {x.size(-1)}"
                )
            x_in = x  # already embedded

        # ---- 3) OPE + PIGL (global_ordering ALWAYS flows through these) ----
        x_ope = self.ope(x_in, global_ordering)           # [B,m,embedding_dim]
        x_pigl = self.pigl(x_ope, importance_scores)      # [B,m,embedding_dim]

        # ---- 4) sequential processor ----
        # We ALWAYS pass global_ordering. The SequentialProcessor* forward() is designed
        # to (a) require it for attention backbones, (b) ignore it for non-attention backbones.
        y_hat = self.processor(x_pigl, global_ordering=global_ordering)

        if return_intermediates:
            return DynaTabOutputs(y_hat=y_hat, x_in=x_in, x_ope=x_ope, x_pigl=x_pigl)
        return y_hat


# ---------------------------
# Public user-facing classes
# ---------------------------
class DynaTabBinary(_DynaTabBase):
    def __init__(
        self,
        num_features: int,
        embedding_dim: int,
        backbone: BackboneName = "Transformer",
        backbone_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            task="binary",
            num_features=num_features,
            embedding_dim=embedding_dim,
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
        )


class DynaTabMulti(_DynaTabBase):
    def __init__(
        self,
        num_features: int,
        embedding_dim: int,
        num_classes: int,
        backbone: BackboneName = "Transformer",
        backbone_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            task="multiclass",
            num_features=num_features,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
        )


class DynaTabRegression(_DynaTabBase):
    def __init__(
        self,
        num_features: int,
        embedding_dim: int,
        backbone: BackboneName = "Transformer",
        backbone_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            task="regression",
            num_features=num_features,
            embedding_dim=embedding_dim,
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
        )