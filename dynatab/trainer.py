# trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import evaluate_predictions
from .customloss import CustomFeatureLoss
from .model import DynaTabBinary, DynaTabMulti, DynaTabRegression, BackboneName, TaskType


# ============================================================
# Configs
# ============================================================
@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 0.0
    device: Optional[torch.device] = None
    print_every: int = 10


@dataclass
class DFOConfig:
    # Matches your reorder_and_evaluate signature
    metric: str = "variance"
    num_clusters: int = 5
    order: str = "ascending"
    mutation_prob: float = 0.5
    tolerance: float = 0.01
    seed: Optional[int] = None


@dataclass
class ModelConfig:
    task: TaskType = "binary"
    backbone: BackboneName = "Transformer"
    embedding_dim: int = 128
    backbone_kwargs: Optional[Dict[str, Any]] = None
    num_classes: Optional[int] = None   # required for multiclass


@dataclass
class LossConfig:
    """
    loss_mode:
      - "standard": use BCEWithLogitsLoss / CrossEntropyLoss / MSELoss
      - "dispersion": use CustomFeatureLoss in "dispersion" mode
      - "DFO": use CustomFeatureLoss in "DFO" mode (dispersion + global penalty)
    """
    loss_mode: str = "DFO"  # "standard" | "dispersion" | "DFO"
    lambda_disp: float = 0.4
    lambda_global: float = 0.3


# ============================================================
# Helpers
# ============================================================
def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _as_long_1d(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dim() != 1:
        raise ValueError(f"Expected [m] tensor, got {tuple(x.shape)}")
    return x.long()


def _compute_class_weights(task: TaskType, y_train: torch.Tensor, num_classes: Optional[int]) -> Optional[torch.Tensor]:
    """
    Returns:
      - binary: [2] float (balanced)
      - multiclass: [C] float (balanced)
      - regression: None
    """
    if task == "regression":
        return None

    y_np = y_train.detach().cpu().numpy().reshape(-1)

    if task == "binary":
        y_np = y_np.astype(int)
        counts = np.bincount(y_np, minlength=2).astype(np.float64)
        counts[counts == 0] = 1.0
        w = (len(y_np) / (2.0 * counts))
        return torch.tensor(w, dtype=torch.float32)

    if task == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes required for multiclass class weights.")
        y_np = y_np.astype(int)
        counts = np.bincount(y_np, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        w = (len(y_np) / (float(num_classes) * counts))
        return torch.tensor(w, dtype=torch.float32)

    raise ValueError(f"Unknown task: {task}")


def _compute_inter_cluster_weights(centroids: Any) -> Dict[Tuple[int, int], float]:
    """
    centroids: torch.Tensor [K, d] (from your DFO reorder_and_evaluate)
    returns: dict over pairs (i,j) -> weight
    """
    if isinstance(centroids, torch.Tensor):
        c = centroids.detach().cpu().numpy()
    else:
        c = np.asarray(centroids)

    K = int(c.shape[0])
    G: Dict[Tuple[int, int], float] = {}
    for i in range(K):
        for j in range(i + 1, K):
            d = float(np.linalg.norm(c[i] - c[j]))
            w = 1.0 / (d + 1e-10)
            G[(i, j)] = w
            G[(j, i)] = w
    return G


def build_model_from_config(model_cfg: ModelConfig, num_features: int) -> nn.Module:
    backbone_kwargs = model_cfg.backbone_kwargs or {}
    if model_cfg.task == "binary":
        return DynaTabBinary(
            num_features=num_features,
            embedding_dim=model_cfg.embedding_dim,
            backbone=model_cfg.backbone,
            backbone_kwargs=backbone_kwargs,
        )
    if model_cfg.task == "multiclass":
        if model_cfg.num_classes is None:
            raise ValueError("ModelConfig.num_classes is required for multiclass.")
        return DynaTabMulti(
            num_features=num_features,
            embedding_dim=model_cfg.embedding_dim,
            num_classes=model_cfg.num_classes,
            backbone=model_cfg.backbone,
            backbone_kwargs=backbone_kwargs,
        )
    if model_cfg.task == "regression":
        return DynaTabRegression(
            num_features=num_features,
            embedding_dim=model_cfg.embedding_dim,
            backbone=model_cfg.backbone,
            backbone_kwargs=backbone_kwargs,
        )
    raise ValueError(f"Unknown task: {model_cfg.task}")


def build_loss(
    task: TaskType,
    loss_cfg: LossConfig,
    class_weights: Optional[torch.Tensor],
    global_ordering_weights: Optional[Dict[Tuple[int, int], torch.Tensor]],
) -> nn.Module:
    mode = (loss_cfg.loss_mode or "standard").lower().strip()

    if mode in ("dispersion", "dfo"):
        return CustomFeatureLoss(
            task=task,
            lambda_disp=loss_cfg.lambda_disp,
            lambda_global=loss_cfg.lambda_global,
            class_weights=class_weights,
            global_ordering_weights=global_ordering_weights,
            loss_mode=("dfo" if mode == "dfo" else "dispersion"),  # safer + consistent
        )

    if mode == "standard":
        if task == "binary":
            return nn.BCEWithLogitsLoss(reduction="mean")
        if task == "multiclass":
            w = class_weights if class_weights is not None else None
            return nn.CrossEntropyLoss(weight=w)
        if task == "regression":
            return nn.MSELoss()
        raise ValueError(f"Unknown task: {task}")

    raise ValueError(f"Unknown loss_mode: {loss_cfg.loss_mode!r}. Use 'standard', 'dispersion', or 'DFO'.")


def _apply_standard_loss_with_class_weights(
    task: TaskType,
    loss_fn: nn.Module,
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    class_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Only needed for "standard" + binary when you want explicit class-weighting as sample weights.
    For multiclass, CrossEntropyLoss already handles weight.
    For regression, no weights.
    """
    if task != "binary":
        return loss_fn(y_hat, y_true)

    y_true_f = y_true.float().view(-1, 1)  # [B,1]
    if class_weights is None:
        return loss_fn(y_hat, y_true_f)

    idx = y_true_f.view(-1).long().clamp(0, 1)
    w = class_weights.to(device=y_hat.device, dtype=torch.float32)[idx].view(-1, 1)  # [B,1]
    per_elem = nn.functional.binary_cross_entropy_with_logits(y_hat, y_true_f, reduction="none")
    return (per_elem * w).mean()


# ============================================================
# DFO → reorder tensors (trainer-level)
# ============================================================
def run_dfo_reorder(
    X_train_df,
    y_train,
    dfo_cfg: DFOConfig,
    reorder_and_evaluate: Callable[..., Tuple[Any, Any, Any, Any]],
):
    """
    reorder_and_evaluate must return:
      Xr (pd.DataFrame), centroids (torch.Tensor), _, global_ordering (list/np/torch)
    """
    kwargs = dict(
        metric=dfo_cfg.metric,
        num_clusters=dfo_cfg.num_clusters,
        order=dfo_cfg.order,
        mutation_prob=dfo_cfg.mutation_prob,
        tolerance=dfo_cfg.tolerance,
    )
    if dfo_cfg.seed is not None:
        kwargs["seed"] = dfo_cfg.seed

    Xr, centroids, misc, global_ordering = reorder_and_evaluate(X_train_df, y_train, **kwargs)
    cols = list(Xr.columns)
    return Xr, cols, centroids, misc, global_ordering


# ============================================================
# Core training (tensor level)
# ============================================================
def train_one_split(
    model: nn.Module,
    task: TaskType,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    global_ordering: torch.Tensor,
    importance_scores: torch.Tensor,
    loss_fn: nn.Module,
    class_weights: Optional[torch.Tensor] = None,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    eval_metrics: Optional[Iterable[str]] = None,
    cfg: TrainConfig = TrainConfig(),
    loss_mode: str = "standard",
) -> Dict[str, float]:
    """
    Works for binary/multiclass/regression.

    Assumptions about model outputs:
      - binary: logits [B,1]
      - multiclass: logits [B,C]
      - regression: [B, out_dim]
    """
    device = cfg.device or _default_device()

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # enforce dtype/device
    X_train = X_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)

    global_ordering = global_ordering.to(device=device, dtype=torch.long)
    importance_scores = importance_scores.to(device=device, dtype=torch.float32)

    if class_weights is not None:
        class_weights = class_weights.to(device=device, dtype=torch.float32)

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    mode = (loss_mode or "standard").lower().strip()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)

            opt.zero_grad()

            # forward API from model.py (model expects global_ordering + importance_scores always)
            y_hat = model(xb, global_ordering, importance_scores)

            if mode in ("dispersion", "dfo"):
                # CustomFeatureLoss expects raw reordered features [B,m] (or [B,m,1])
                if xb.dim() == 3 and xb.size(-1) == 1:
                    xb_feat = xb.squeeze(-1)
                elif xb.dim() == 2:
                    xb_feat = xb
                else:
                    raise ValueError(
                        f"CustomFeatureLoss expects raw reordered features [B,m] (or [B,m,1]), got {tuple(xb.shape)}"
                    )

                loss = loss_fn(y_hat, yb, xb_feat)
            else:
                loss = _apply_standard_loss_with_class_weights(task, loss_fn, y_hat, yb, class_weights)

            loss.backward()
            opt.step()

            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        if cfg.print_every and (epoch % cfg.print_every == 0 or epoch == 1 or epoch == cfg.epochs):
            print(f"Epoch {epoch:03d} | loss={total / max(n, 1):.6f}")

            if X_val is not None and y_val is not None:
                metrics = evaluate_split(
                    model=model,
                    task=task,
                    X=X_val,
                    y=y_val,
                    global_ordering=global_ordering,
                    importance_scores=importance_scores,
                    metrics=eval_metrics,
                    device=device,
                )
                print("  val:", {k: round(v, 4) for k, v in metrics.items()})

    # final eval on val if provided else train
    if X_val is not None and y_val is not None:
        return evaluate_split(model, task, X_val, y_val, global_ordering, importance_scores, eval_metrics, device=device)
    return evaluate_split(model, task, X_train, y_train, global_ordering, importance_scores, eval_metrics, device=device)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    task: TaskType,
    X: torch.Tensor,
    y: torch.Tensor,
    global_ordering: torch.Tensor,
    importance_scores: torch.Tensor,
    metrics: Optional[Iterable[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or _default_device()
    model.eval()

    X = X.to(device=device, dtype=torch.float32)
    y = y.to(device=device)

    global_ordering = global_ordering.to(device=device, dtype=torch.long)
    importance_scores = importance_scores.to(device=device, dtype=torch.float32)

    y_hat = model(X, global_ordering, importance_scores)

    # For binary AUC/F1 etc, evaluator expects probabilities
    if task == "binary":
        y_hat = torch.sigmoid(y_hat)

    return evaluate_predictions(task=task, y_true=y, y_hat=y_hat, metrics=metrics)


# ============================================================
# High-level “DFO + model + loss” training from DataFrames
# ============================================================
def fit_from_dataframes_one_split(
    *,
    task: TaskType,
    backbone: BackboneName,
    embedding_dim: int,
    X_train_df,
    y_train,
    X_val_df=None,
    y_val=None,
    X_test_df=None,
    y_test=None,
    dfo_cfg: DFOConfig,
    reorder_and_evaluate: Callable[..., Tuple[Any, Any, Any, Any]],
    backbone_kwargs: Optional[Dict[str, Any]] = None,
    num_classes: Optional[int] = None,
    loss_cfg: LossConfig = LossConfig(),
    cfg: TrainConfig = TrainConfig(),
    eval_metrics: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    device = cfg.device or _default_device()

    # ---- 1) DFO reorder (train only) ----
    Xr, cols, centroids, _, global_ordering = run_dfo_reorder(X_train_df, y_train, dfo_cfg, reorder_and_evaluate)

    if X_val_df is not None:
        X_val_df = X_val_df[cols]
    if X_test_df is not None:
        X_test_df = X_test_df[cols]

    # ---- 2) tensors ----
    X_train_t = torch.tensor(np.asarray(Xr.values), dtype=torch.float32)
    if task == "binary":
        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)
    elif task == "multiclass":
        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.long).view(-1)
    else:
        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)

    X_val_t = y_val_t = None
    if X_val_df is not None and y_val is not None:
        X_val_t = torch.tensor(np.asarray(X_val_df.values), dtype=torch.float32)
        if task == "binary":
            y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).view(-1, 1)
        elif task == "multiclass":
            y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.long).view(-1)
        else:
            y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).view(-1, 1)

    X_test_t = y_test_t = None
    if X_test_df is not None and y_test is not None:
        X_test_t = torch.tensor(np.asarray(X_test_df.values), dtype=torch.float32)
        if task == "binary":
            y_test_t = torch.tensor(np.asarray(y_test), dtype=torch.float32).view(-1, 1)
        elif task == "multiclass":
            y_test_t = torch.tensor(np.asarray(y_test), dtype=torch.long).view(-1)
        else:
            y_test_t = torch.tensor(np.asarray(y_test), dtype=torch.float32).view(-1, 1)

    m = int(X_train_t.shape[1])
    global_ordering_t = _as_long_1d(torch.tensor(np.asarray(global_ordering)))
    if global_ordering_t.numel() != m:
        raise ValueError(f"global_ordering length mismatch: got {global_ordering_t.numel()} but m={m}")

    # ---- 3) importance scores ----
    importance_scores = torch.var(X_train_t, dim=0).detach()

    # ---- 4) class weights ----
    class_weights = _compute_class_weights(task, y_train_t, num_classes)

    # ---- 5) global ordering weights ----
    gow_np = _compute_inter_cluster_weights(centroids)
    global_ordering_weights = {k: torch.tensor(v, dtype=torch.float32) for k, v in gow_np.items()}

    # ---- 6) model ----
    model_cfg = ModelConfig(
        task=task,
        backbone=backbone,
        embedding_dim=embedding_dim,
        backbone_kwargs=backbone_kwargs or {},
        num_classes=num_classes,
    )
    model = build_model_from_config(model_cfg, num_features=m)

    # ---- 7) loss ----
    loss_fn = build_loss(
        task=task,
        loss_cfg=loss_cfg,
        class_weights=class_weights,
        global_ordering_weights=global_ordering_weights,
    )

    # ---- 8) train ----
    train_metrics = train_one_split(
        model=model,
        task=task,
        X_train=X_train_t,
        y_train=y_train_t,
        global_ordering=global_ordering_t,
        importance_scores=importance_scores,
        loss_fn=loss_fn,
        class_weights=class_weights,
        X_val=X_val_t,
        y_val=y_val_t,
        eval_metrics=eval_metrics,
        cfg=cfg,
        loss_mode=loss_cfg.loss_mode,
    )

    out: Dict[str, Any] = dict(
        model=model.to(device),
        cols=cols,
        global_ordering=global_ordering_t.to(device),
        importance_scores=importance_scores.to(device),
        train_or_val_metrics=train_metrics,
    )

    if X_test_t is not None and y_test_t is not None:
        test_metrics = evaluate_split(
            model=model,
            task=task,
            X=X_test_t,
            y=y_test_t,
            global_ordering=global_ordering_t,
            importance_scores=importance_scores,
            metrics=eval_metrics,
            device=device,
        )
        out["test_metrics"] = test_metrics

    return out


# ============================================================
# CV runner (keeps DFO fixed per fold, or recompute per fold)
# ============================================================
def cross_validate(
    *,
    task: TaskType,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    cfg: TrainConfig,
    X: torch.Tensor,
    y: torch.Tensor,
    folds: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    global_ordering: torch.Tensor,
    importance_scores: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    global_ordering_weights: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    eval_metrics: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    k = 0

    for train_idx, val_idx in folds:
        train_idx = torch.as_tensor(train_idx, dtype=torch.long)
        val_idx = torch.as_tensor(val_idx, dtype=torch.long)

        model = build_model_from_config(model_cfg, num_features=int(X.shape[1]))
        loss_fn = build_loss(task, loss_cfg, class_weights, global_ordering_weights)

        metrics_out = train_one_split(
            model=model,
            task=task,
            X_train=X[train_idx],
            y_train=y[train_idx],
            global_ordering=global_ordering,
            importance_scores=importance_scores,
            loss_fn=loss_fn,
            class_weights=class_weights,
            X_val=X[val_idx],
            y_val=y[val_idx],
            eval_metrics=eval_metrics,
            cfg=cfg,
            loss_mode=loss_cfg.loss_mode,
        )

        for key, val in metrics_out.items():
            acc[key] = acc.get(key, 0.0) + float(val)
        k += 1

    return {k_: v_ / max(k, 1) for k_, v_ in acc.items()}