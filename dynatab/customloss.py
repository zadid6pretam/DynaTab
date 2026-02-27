# DynaTab unified loss with REQUIRED class-imbalance handling for:
#   - binary classification
#   - multiclass classification
#   - regression (no class weights)
#
# Implements:
#   - base task loss (weighted BCE / weighted CE / MSE)
#   - + dispersion penalty
#   - + global-order penalty (DFO mode)
#
from __future__ import annotations

from typing import Dict, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

TaskType = Literal["binary", "multiclass", "regression"]
LossMode = Literal["base", "dispersion", "DFO"]


def compute_class_weights_from_labels(
    y: torch.Tensor,
    num_classes: Optional[int] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from labels.

    Args:
        y: [B] integer labels (0..C-1) for multiclass OR {0,1} for binary.
        num_classes: if None, inferred as max(y)+1.
        normalize: if True, weights are scaled to have mean 1.0.
    Returns:
        weights: Tensor[C] float32
    """
    if y.dim() != 1:
        y = y.view(-1)
    y = y.long()

    C = int(num_classes) if num_classes is not None else int(torch.max(y).item()) + 1
    if C <= 1:
        return torch.ones(1, dtype=torch.float32, device=y.device)

    counts = torch.bincount(y.clamp(min=0, max=C - 1), minlength=C).float()  # [C]
    inv = 1.0 / (counts + eps)
    weights = inv

    if normalize:
        weights = weights / (weights.mean() + eps)  # mean ~ 1
    return weights.float()


class DynaTabCustomLoss(nn.Module):
    """
    Args:
        task: "binary" | "multiclass" | "regression"
        loss_mode: "base" | "dispersion" | "DFO"
        lambda_disp: weight for dispersion penalty
        lambda_global: weight for global-order penalty (only in DFO)
        from_logits:
            - binary: True -> BCEWithLogitsLoss; False -> BCE on probabilities
            - multiclass: True recommended (CE expects logits). If False, we log() probs.
        class_weights:
            - REQUIRED for binary/multiclass imbalance handling.
              If None, you can set auto_class_weights=True to compute from y_true each forward.
            - binary: Tensor[2]
            - multiclass: Tensor[C]
        auto_class_weights:
            - If True and class_weights is None, compute weights from the current y_true batch.
              (For stable training, you typically compute once on the training set and pass it in.)
        global_ordering_weights: dict[(u,v)->w] for global penalty (optional)
    """

    def __init__(
        self,
        task: TaskType,
        loss_mode: LossMode = "DFO",
        lambda_disp: float = 0.4,
        lambda_global: float = 0.3,
        from_logits: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        auto_class_weights: bool = False,
        num_classes: Optional[int] = None,  # for multiclass auto weighting
        global_ordering_weights: Optional[Dict[Tuple[int, int], float]] = None,
        reduction: Literal["mean", "sum"] = "mean",
        eps: float = 1e-12,
    ):
        super().__init__()
        self.task: TaskType = task
        self.loss_mode: LossMode = loss_mode
        self.lambda_disp = float(lambda_disp)
        self.lambda_global = float(lambda_global)
        self.from_logits = bool(from_logits)
        self.auto_class_weights = bool(auto_class_weights)
        self.num_classes = num_classes
        self.global_ordering_weights = global_ordering_weights or {}
        self.reduction = reduction
        self.eps = float(eps)

        if self.lambda_disp < 0 or self.lambda_global < 0:
            raise ValueError("lambda_disp and lambda_global must be non-negative.")
        if self.loss_mode == "DFO" and (self.lambda_disp + self.lambda_global) >= 1.0 + 1e-9:
            raise ValueError("For DFO, require lambda_disp + lambda_global < 1.")
        if self.loss_mode == "dispersion" and self.lambda_disp >= 1.0 + 1e-9:
            raise ValueError("For dispersion, require lambda_disp < 1.")

        # store fixed weights (optional)
        if class_weights is not None:
            if not torch.is_tensor(class_weights):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None  # type: ignore[assignment]

    # -----------------------
    # X penalties
    # -----------------------
    @staticmethod
    def _ensure_Bm(X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2:
            return X
        if X.dim() == 3:
            return X.mean(dim=-1)
        raise ValueError(f"X must be [B,m] or [B,m,d]. Got {tuple(X.shape)}")

    def dispersion_penalty(self, X: torch.Tensor) -> torch.Tensor:
        X1 = self._ensure_Bm(X)
        if X1.size(1) < 2:
            return X1.new_tensor(0.0)
        dist = torch.abs(X1[:, :-1] - X1[:, 1:]).sum(dim=1)  # [B]
        return dist.mean() if self.reduction == "mean" else dist.sum()

    def global_order_penalty(self, X: torch.Tensor) -> torch.Tensor:
        X1 = self._ensure_Bm(X)
        if not self.global_ordering_weights:
            return X1.new_tensor(0.0)

        m = X1.size(1)
        terms = []
        for (u, v), w in self.global_ordering_weights.items():
            if not (0 <= u < m and 0 <= v < m):
                raise ValueError(f"global_ordering_weights key {(u,v)} out of bounds for m={m}.")
            w_t = X1.new_tensor(float(w))
            terms.append(torch.abs(X1[:, u] - X1[:, v]).mean() * w_t)

        if not terms:
            return X1.new_tensor(0.0)
        stacked = torch.stack(terms)
        return stacked.mean() if self.reduction == "mean" else stacked.sum()

    # -----------------------
    # class weights resolution
    # -----------------------
    def _require_or_compute_weights(self, y_true_labels: torch.Tensor) -> torch.Tensor:
        """
        Returns Tensor[C] weights for classification tasks.
        Enforces that imbalance handling exists:
          - use provided self.class_weights, OR
          - auto compute if auto_class_weights=True
        """
        if self.class_weights is not None:
            return self.class_weights.to(device=y_true_labels.device)

        if not self.auto_class_weights:
            raise ValueError(
                "class_weights is required for class-imbalance handling. "
                "Provide class_weights=Tensor[C] OR set auto_class_weights=True."
            )

        # auto compute from labels
        return compute_class_weights_from_labels(
            y_true_labels,
            num_classes=self.num_classes,
            normalize=True,
            eps=self.eps,
        ).to(device=y_true_labels.device)

    # -----------------------
    # task losses (weighted)
    # -----------------------
    def _binary_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_true -> [B] long labels for weighting + [B,1] float for BCE
        if y_true.dim() == 2 and y_true.size(1) == 1:
            y_true_f = y_true.float()
            y_lbl = y_true.squeeze(1).long()
        elif y_true.dim() == 1:
            y_true_f = y_true.unsqueeze(1).float()
            y_lbl = y_true.long()
        else:
            raise ValueError(f"[binary] y_true must be [B] or [B,1]. Got {tuple(y_true.shape)}")

        # y_pred -> [B,1]
        if y_pred.dim() == 1:
            y_pred_ = y_pred.unsqueeze(1)
        elif y_pred.dim() == 2 and y_pred.size(1) == 1:
            y_pred_ = y_pred
        else:
            raise ValueError(f"[binary] y_pred must be [B] or [B,1]. Got {tuple(y_pred.shape)}")

        # class weights (required or auto)
        cw = self._require_or_compute_weights(y_lbl)  # Tensor[2]
        cw = cw.to(device=y_lbl.device)
        if cw.numel() < 2:
            raise ValueError("[binary] class_weights must have 2 entries for classes {0,1}.")
        y_lbl = y_lbl.clamp(0, 1)
        sample_w = cw[y_lbl].unsqueeze(1)  # [B,1]

        if self.from_logits:
            return F.binary_cross_entropy_with_logits(
                y_pred_, y_true_f, weight=sample_w, reduction=self.reduction
            )
        return F.binary_cross_entropy(
            y_pred_, y_true_f, weight=sample_w, reduction=self.reduction
        )

    def _multiclass_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_true -> [B] long
        if y_true.dim() != 1:
            raise ValueError(f"[multiclass] y_true must be [B] class indices. Got {tuple(y_true.shape)}")
        y_lbl = y_true.long()

        # y_pred -> [B,C]
        if y_pred.dim() != 2:
            raise ValueError(f"[multiclass] y_pred must be [B,C]. Got {tuple(y_pred.shape)}")
        C = y_pred.size(1)

        # weights (required or auto)
        cw = self._require_or_compute_weights(y_lbl)  # Tensor[C]
        cw = cw.to(device=y_pred.device)
        if cw.numel() != C:
            raise ValueError(f"[multiclass] class_weights length {cw.numel()} != num classes {C}.")

        # CE expects logits
        logits = y_pred
        if not self.from_logits:
            probs = torch.clamp(y_pred, min=self.eps, max=1.0 - self.eps)
            logits = torch.log(probs)

        return F.cross_entropy(logits, y_lbl, weight=cw, reduction=self.reduction)

    def _regression_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # allow [B] or [B,1]
        if y_true.dim() == 2 and y_true.size(1) == 1:
            y_true_ = y_true.squeeze(1)
        elif y_true.dim() == 1:
            y_true_ = y_true
        else:
            raise ValueError(f"[regression] y_true must be [B] or [B,1]. Got {tuple(y_true.shape)}")

        if y_pred.dim() == 2 and y_pred.size(1) == 1:
            y_pred_ = y_pred.squeeze(1)
        elif y_pred.dim() == 1:
            y_pred_ = y_pred
        else:
            raise ValueError(f"[regression] y_pred must be [B] or [B,1]. Got {tuple(y_pred.shape)}")

        return F.mse_loss(y_pred_.float(), y_true_.float(), reduction=self.reduction)

    def task_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.task == "binary":
            return self._binary_loss(y_pred, y_true)
        if self.task == "multiclass":
            return self._multiclass_loss(y_pred, y_true)
        if self.task == "regression":
            return self._regression_loss(y_pred, y_true)
        raise ValueError(f"Unknown task: {self.task}")

    # -----------------------
    # forward
    # -----------------------
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, X_train_reordered: torch.Tensor) -> torch.Tensor:
        base = self.task_loss(y_pred, y_true)

        if self.loss_mode == "base":
            return base

        disp = self.dispersion_penalty(X_train_reordered)

        if self.loss_mode == "dispersion":
            return (self.lambda_disp * disp) + ((1.0 - self.lambda_disp) * base)

        glob = self.global_order_penalty(X_train_reordered)
        coef = max(0.0, 1.0 - self.lambda_disp - self.lambda_global)
        return (self.lambda_disp * disp) + (self.lambda_global * glob) + (coef * base)

# ---- Aliases for package API compatibility ----
CustomFeatureLoss = DynaTabCustomLoss
__all__ = ["DynaTabCustomLoss", "CustomFeatureLoss"]