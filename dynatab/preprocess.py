# dynatab/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class Standardizer:
    """Simple column-wise standardizer fit on TRAIN only."""
    mu: pd.Series
    sd: pd.Series

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Align columns; assume X already has same columns as mu/sd.
        return (X - self.mu) / self.sd


def _to_dataframe(X: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array-like, got shape {X.shape}")
    if columns is None:
        columns = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns)


def sanitize_numeric_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf, keep numeric only (or raise if non-numeric present)."""
    X = X.copy()
    # If non-numeric columns exist, try converting; otherwise raise to avoid silent bugs.
    non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_num:
        # try best-effort conversion
        for c in non_num:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def impute_median_train(
    X_train: pd.DataFrame,
    X_other: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.Series]:
    """
    Median imputation fit on TRAIN only; applies to X_other if provided.
    Returns imputed train, imputed other, and medians.
    """
    X_train = X_train.copy()
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    if X_other is not None:
        X_other = X_other.copy().fillna(med)
    return X_train, X_other, med


def fit_standardizer(X_train: pd.DataFrame) -> Standardizer:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0).replace(0, 1.0)
    return Standardizer(mu=mu, sd=sd)


def train_transform_preprocess(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    columns: Optional[list] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Standardizer, pd.Series]:
    """
    Convenience: convert -> sanitize -> impute (train medians) -> standardize (train stats).
    Returns (X_train_proc, X_val_proc, standardizer, medians)
    """
    Xtr = _to_dataframe(X_train, columns=columns)
    Xva = _to_dataframe(X_val, columns=columns) if X_val is not None else None

    Xtr = sanitize_numeric_dataframe(Xtr)
    if Xva is not None:
        Xva = sanitize_numeric_dataframe(Xva)

    Xtr, Xva, med = impute_median_train(Xtr, Xva)
    std = fit_standardizer(Xtr)

    Xtr = std.transform(Xtr)
    if Xva is not None:
        # align just in case
        Xva = Xva[Xtr.columns]
        Xva = std.transform(Xva)

    return Xtr, Xva, std, med