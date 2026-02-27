# dynatab/estimators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .trainer import (
    TrainConfig,
    DFOConfig,
    ModelConfig,
    LossConfig,
    build_model_from_config,
    build_loss,
    train_one_split,
    evaluate_split,
    _compute_class_weights,      # relies on your trainer.py; keep if you’re okay importing internals
    _compute_inter_cluster_weights,  # same note
)
from .preprocess import train_transform_preprocess, _to_dataframe, sanitize_numeric_dataframe
from .dfo import reorder_and_evaluate  # your DFO entrypoint


ArrayLikeX = Union[pd.DataFrame, np.ndarray]
ArrayLikeY = Union[pd.Series, np.ndarray, list]


def _default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _infer_num_classes(y: np.ndarray) -> int:
    uniq = np.unique(y.astype(int))
    return int(len(uniq))


def _ensure_1d_np(y: ArrayLikeY) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = np.asarray(y).reshape(-1)
    else:
        y = np.asarray(y).reshape(-1)
    return y


def _to_task_y_tensor(task: str, y_np: np.ndarray) -> torch.Tensor:
    if task == "binary":
        # expects [B,1] float
        return torch.tensor(y_np.astype(np.float32), dtype=torch.float32).view(-1, 1)
    if task == "multiclass":
        return torch.tensor(y_np.astype(np.int64), dtype=torch.long).view(-1)
    if task == "regression":
        return torch.tensor(y_np.astype(np.float32), dtype=torch.float32).view(-1, 1)
    raise ValueError(f"Unknown task: {task}")


@torch.no_grad()
def _predict_logits(
    model: torch.nn.Module,
    X: torch.Tensor,
    global_ordering: torch.Tensor,
    importance_scores: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    X = X.to(device=device, dtype=torch.float32)
    global_ordering = global_ordering.to(device=device, dtype=torch.long)
    importance_scores = importance_scores.to(device=device, dtype=torch.float32)
    out = model(X, global_ordering, importance_scores)
    return out


@dataclass
class _Artifacts:
    cols: list
    standardizer_mu: pd.Series
    standardizer_sd: pd.Series
    medians: pd.Series
    global_ordering: torch.Tensor
    importance_scores: torch.Tensor
    global_ordering_weights: Optional[Dict[Tuple[int, int], torch.Tensor]] = None


class DynaTabEstimatorBase:
    """
    sklearn-like estimator wrapper over your existing trainer + model stack.

    - End-to-end training (OPE/PIGL/DMA/backbone all trainable).
    - DFO is run once inside fit() (or you can choose to re-run it per fold externally).
    - Stores preprocessing + ordering artifacts for consistent inference.
    """

    def __init__(
        self,
        *,
        task: str,
        backbone: str = "Transformer",
        embedding_dim: int = 128,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        dfo_cfg: Optional[DFOConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
        loss_cfg: Optional[LossConfig] = None,
        num_classes: Optional[int] = None,  # for multiclass
        eval_metrics: Optional[Iterable[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        standardize: bool = True,
    ):
        self.task = task
        self.backbone = backbone
        self.embedding_dim = int(embedding_dim)
        self.backbone_kwargs = backbone_kwargs or {}
        self.dfo_cfg = dfo_cfg or DFOConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.loss_cfg = loss_cfg or LossConfig()
        self.num_classes = num_classes
        self.eval_metrics = list(eval_metrics) if eval_metrics is not None else None
        self.device = _default_device(device)
        self.standardize = bool(standardize)

        # Fit outputs
        self.model_: Optional[torch.nn.Module] = None
        self.artifacts_: Optional[_Artifacts] = None
        self.train_metrics_: Optional[Dict[str, float]] = None

    # --------------------
    # Preprocess helpers
    # --------------------
    def _fit_preprocess(self, X_train: ArrayLikeX, X_val: Optional[ArrayLikeX] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.Series, pd.Series, pd.Series]:
        Xtr = _to_dataframe(X_train)
        Xva = _to_dataframe(X_val, columns=list(Xtr.columns)) if X_val is not None else None

        Xtr = sanitize_numeric_dataframe(Xtr)
        if Xva is not None:
            Xva = sanitize_numeric_dataframe(Xva)

        # impute + standardize on train only
        if self.standardize:
            Xtr_p, Xva_p, std, med = train_transform_preprocess(Xtr, Xva, columns=list(Xtr.columns))
            return Xtr_p, Xva_p, std.mu, std.sd, med

        # if not standardizing, still impute train medians
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)
        if Xva is not None:
            Xva = Xva.fillna(med)
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0).replace(0, 1.0)
        return Xtr, Xva, mu, sd, med

    def _transform_preprocess(self, X: ArrayLikeX) -> pd.DataFrame:
        if self.artifacts_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        Xdf = _to_dataframe(X, columns=self.artifacts_.cols)
        Xdf = sanitize_numeric_dataframe(Xdf)
        # align
        Xdf = Xdf[self.artifacts_.cols]
        # impute with train medians
        Xdf = Xdf.fillna(self.artifacts_.medians)
        # standardize with train stats if enabled
        if self.standardize:
            Xdf = (Xdf - self.artifacts_.standardizer_mu) / self.artifacts_.standardizer_sd
        return Xdf

    # --------------------
    # Core API
    # --------------------
    def fit(self, X: ArrayLikeX, y: ArrayLikeY, X_val: Optional[ArrayLikeX] = None, y_val: Optional[ArrayLikeY] = None) -> "DynaTabEstimatorBase":
        # 0) preprocess
        Xtr_p, Xva_p, mu, sd, med = self._fit_preprocess(X, X_val)

        y_np = _ensure_1d_np(y)

        # Infer num_classes if multiclass
        if self.task == "multiclass" and self.num_classes is None:
            self.num_classes = _infer_num_classes(y_np)

        # 1) DFO reorder on TRAIN only
        Xr, centroids, _, global_ordering = reorder_and_evaluate(
            Xtr_p,
            y_np,
            metric=self.dfo_cfg.metric,
            num_clusters=int(self.dfo_cfg.num_clusters),
            order=self.dfo_cfg.order,
            mutation_prob=float(self.dfo_cfg.mutation_prob),
            tolerance=float(self.dfo_cfg.tolerance),
            **({"seed": self.dfo_cfg.seed} if self.dfo_cfg.seed is not None else {}),
        )
        cols = list(Xr.columns)

        # Reorder val using TRAIN-derived cols
        if Xva_p is not None:
            Xva_p = Xva_p[cols]

        # 2) tensors
        X_train_t = torch.tensor(np.asarray(Xr.values), dtype=torch.float32)
        y_train_t = _to_task_y_tensor(self.task, y_np)

        X_val_t = y_val_t = None
        if Xva_p is not None and y_val is not None:
            yva_np = _ensure_1d_np(y_val)
            X_val_t = torch.tensor(np.asarray(Xva_p.values), dtype=torch.float32)
            y_val_t = _to_task_y_tensor(self.task, yva_np)

        m = int(X_train_t.shape[1])

        # Important: after reorder, global ordering positions should align to the reordered features.
        # If your global_ordering returned by DFO is already "positional indices for OPE", keep it.
        # If it's a permutation mapping, you may prefer identity here.
        global_ordering_t = torch.tensor(np.asarray(global_ordering), dtype=torch.long).view(-1)
        if global_ordering_t.numel() != m:
            # fallback: identity (safe after reorder)
            global_ordering_t = torch.arange(m, dtype=torch.long)

        # 3) importance + weights
        importance_scores = torch.var(X_train_t, dim=0).detach()

        class_weights = _compute_class_weights(self.task, y_train_t, self.num_classes)
        gow_np = _compute_inter_cluster_weights(centroids) if centroids is not None else {}
        global_ordering_weights = {k: torch.tensor(v, dtype=torch.float32) for k, v in gow_np.items()} if gow_np else None

        # 4) build model + loss
        model_cfg = ModelConfig(
            task=self.task,
            backbone=self.backbone,
            embedding_dim=self.embedding_dim,
            backbone_kwargs=self.backbone_kwargs,
            num_classes=self.num_classes,
        )
        model = build_model_from_config(model_cfg, num_features=m)

        loss_fn = build_loss(
            task=self.task,
            loss_cfg=self.loss_cfg,
            class_weights=class_weights,
            global_ordering_weights=global_ordering_weights,
        )

        # 5) train
        cfg = self.train_cfg
        cfg.device = self.device  # ensure device
        metrics_out = train_one_split(
            model=model,
            task=self.task,
            X_train=X_train_t,
            y_train=y_train_t,
            global_ordering=global_ordering_t,
            importance_scores=importance_scores,
            loss_fn=loss_fn,
            class_weights=class_weights,
            X_val=X_val_t,
            y_val=y_val_t,
            eval_metrics=self.eval_metrics,
            cfg=cfg,
            loss_mode=self.loss_cfg.loss_mode,
        )

        # Store
        self.model_ = model.to(self.device)
        self.artifacts_ = _Artifacts(
            cols=cols,
            standardizer_mu=mu[cols],
            standardizer_sd=sd[cols].replace(0, 1.0),
            medians=med[cols],
            global_ordering=global_ordering_t.to(self.device),
            importance_scores=importance_scores.to(self.device),
            global_ordering_weights=global_ordering_weights,
        )
        self.train_metrics_ = metrics_out
        return self

    def _prepare_X_tensor(self, X: ArrayLikeX) -> torch.Tensor:
        Xdf = self._transform_preprocess(X)
        # reorder
        Xdf = Xdf[self.artifacts_.cols]  # type: ignore
        return torch.tensor(np.asarray(Xdf.values), dtype=torch.float32)

    def predict(self, X: ArrayLikeX) -> np.ndarray:
        if self.model_ is None or self.artifacts_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        Xt = self._prepare_X_tensor(X)
        logits = _predict_logits(
            self.model_,
            Xt,
            self.artifacts_.global_ordering,
            self.artifacts_.importance_scores,
            self.device,
        )

        if self.task == "binary":
            proba = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
            return (proba >= 0.5).astype(int)

        if self.task == "multiclass":
            pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            return pred.astype(int)

        if self.task == "regression":
            return logits.view(-1).detach().cpu().numpy()

        raise ValueError(f"Unknown task: {self.task}")

    def predict_proba(self, X: ArrayLikeX) -> np.ndarray:
        if self.task not in ("binary", "multiclass"):
            raise ValueError("predict_proba is only available for binary/multiclass tasks.")
        if self.model_ is None or self.artifacts_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        Xt = self._prepare_X_tensor(X)
        logits = _predict_logits(
            self.model_,
            Xt,
            self.artifacts_.global_ordering,
            self.artifacts_.importance_scores,
            self.device,
        )

        if self.task == "binary":
            proba1 = torch.sigmoid(logits).view(-1, 1)
            proba0 = 1.0 - proba1
            return torch.cat([proba0, proba1], dim=1).detach().cpu().numpy()

        # multiclass
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return probs

    def score(self, X: ArrayLikeX, y: ArrayLikeY, metrics: Optional[Iterable[str]] = None) -> Dict[str, float]:
        if self.model_ is None or self.artifacts_ is None:
            raise RuntimeError("Estimator is not fit yet.")

        y_np = _ensure_1d_np(y)
        Xt = self._prepare_X_tensor(X)
        yt = _to_task_y_tensor(self.task, y_np)

        return evaluate_split(
            model=self.model_,
            task=self.task,
            X=Xt,
            y=yt,
            global_ordering=self.artifacts_.global_ordering,
            importance_scores=self.artifacts_.importance_scores,
            metrics=list(metrics) if metrics is not None else self.eval_metrics,
            device=self.device,
        )


# ---------------------------
# Public concrete estimators
# ---------------------------
class DynaTabClassifier(DynaTabEstimatorBase):
    """
    Use for:
      - binary: task="binary"
      - multiclass: task="multiclass" (set num_classes or inferred from y)
    """

    def __init__(self, *, task: str = "binary", **kwargs):
        if task not in ("binary", "multiclass"):
            raise ValueError("DynaTabClassifier supports task='binary' or 'multiclass'.")
        super().__init__(task=task, **kwargs)


class DynaTabRegressor(DynaTabEstimatorBase):
    def __init__(self, **kwargs):
        super().__init__(task="regression", **kwargs)