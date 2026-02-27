# metrics.py
from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional

import torch

TaskType = Literal["binary", "multiclass", "regression"]


@torch.no_grad()
def _binary_confusion(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_true,y_pred: [N] int {0,1}
    tp = torch.sum((y_true == 1) & (y_pred == 1)).item()
    tn = torch.sum((y_true == 0) & (y_pred == 0)).item()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).item()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).item()
    return tp, tn, fp, fn


@torch.no_grad()
def _precision_recall_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


@torch.no_grad()
def evaluate_predictions(
    task: TaskType,
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
    metrics: Optional[Iterable[str]] = None,
    threshold: float = 0.5,
    average: Literal["macro", "micro", "weighted"] = "macro",
) -> Dict[str, float]:
    """
    Flexible evaluation over already-computed model outputs.

    Args:
      task:
        - "binary": y_hat can be probs [N] or [N,1] or logits (user decides; see note below)
        - "multiclass": y_hat should be logits/probs [N,C]
        - "regression": y_hat should be [N] or [N,1]
      y_true:
        - binary: [N] {0,1} (int/float ok)
        - multiclass: [N] class indices (int)
        - regression: [N] float
      metrics: list like ["acc","f1","precision","recall","auc"] or ["mse","r2"]
               If None, chooses a sensible default per task.
      threshold: for binary conversion prob->label
      average: for multiclass f1/precision/recall aggregation (macro/micro/weighted)

    Returns: dict metric_name -> float

    Note:
      - AUC uses sklearn if installed; otherwise raises an ImportError.
      - For binary AUC, y_hat must represent probabilities (or monotonic scores).
    """
    if metrics is None:
        metrics = (
            ["acc", "auc", "f1", "precision", "recall"]
            if task == "binary"
            else (["acc", "f1_macro"] if task == "multiclass" else ["mse", "r2"])
        )
    metrics = list(metrics)

    # flatten shapes
    y_true = y_true.detach()
    y_hat = y_hat.detach()

    out: Dict[str, float] = {}

    if task == "binary":
        yt = y_true.view(-1).long()
        yh = y_hat.view(-1)

        # interpret yh as probabilities in [0,1] if possible; otherwise user can pass sigmoid(y_hat)
        y_prob = yh
        y_pred = (y_prob >= threshold).long()

        if "acc" in metrics:
            out["acc"] = (y_pred == yt).float().mean().item()

        if any(m in metrics for m in ("precision", "recall", "f1")):
            tp, tn, fp, fn = _binary_confusion(yt, y_pred)
            prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
            if "precision" in metrics:
                out["precision"] = float(prec)
            if "recall" in metrics:
                out["recall"] = float(rec)
            if "f1" in metrics:
                out["f1"] = float(f1)

        if "auc" in metrics:
            try:
                from sklearn.metrics import roc_auc_score  # type: ignore
            except Exception as e:
                raise ImportError("AUC requires scikit-learn: pip install scikit-learn") from e
            out["auc"] = float(roc_auc_score(yt.cpu().numpy(), y_prob.cpu().numpy()))

        return out

    if task == "multiclass":
        yt = y_true.view(-1).long()
        if y_hat.dim() != 2:
            raise ValueError(f"multiclass expects y_hat [N,C]. Got {tuple(y_hat.shape)}")
        y_pred = torch.argmax(y_hat, dim=1)

        if "acc" in metrics:
            out["acc"] = (y_pred == yt).float().mean().item()

        # F1/precision/recall via sklearn (much less error-prone for macro/micro/weighted)
        needs_sklearn = any(
            m in metrics for m in ("f1", "f1_macro", "f1_micro", "precision", "recall", "precision_macro", "recall_macro")
        )
        if needs_sklearn:
            try:
                from sklearn.metrics import f1_score, precision_score, recall_score  # type: ignore
            except Exception as e:
                raise ImportError("multiclass F1/precision/recall require scikit-learn: pip install scikit-learn") from e

            y_np = yt.cpu().numpy()
            p_np = y_pred.cpu().numpy()

            if "f1" in metrics:
                out["f1"] = float(f1_score(y_np, p_np, average=average))
            if "f1_macro" in metrics:
                out["f1_macro"] = float(f1_score(y_np, p_np, average="macro"))
            if "f1_micro" in metrics:
                out["f1_micro"] = float(f1_score(y_np, p_np, average="micro"))
            if "precision" in metrics:
                out["precision"] = float(precision_score(y_np, p_np, average=average, zero_division=0))
            if "recall" in metrics:
                out["recall"] = float(recall_score(y_np, p_np, average=average, zero_division=0))
            if "precision_macro" in metrics:
                out["precision_macro"] = float(precision_score(y_np, p_np, average="macro", zero_division=0))
            if "recall_macro" in metrics:
                out["recall_macro"] = float(recall_score(y_np, p_np, average="macro", zero_division=0))

        return out

    if task == "regression":
        yt = y_true.view(-1).float()
        yh = y_hat.view(-1).float()

        if "mse" in metrics:
            out["mse"] = torch.mean((yh - yt) ** 2).item()
        if "mae" in metrics:
            out["mae"] = torch.mean(torch.abs(yh - yt)).item()
        if "r2" in metrics:
            # R^2 = 1 - SSE/SST
            sse = torch.sum((yh - yt) ** 2)
            sst = torch.sum((yt - torch.mean(yt)) ** 2) + 1e-12
            out["r2"] = (1.0 - (sse / sst)).item()

        return out

    raise ValueError(f"Unknown task: {task}")