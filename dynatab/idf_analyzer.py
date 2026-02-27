# dynatab/idf_analyzer.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


ArrayLike = Union[np.ndarray, pd.DataFrame]


def _to_numpy(X: ArrayLike) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    else:
        X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape={X.shape}")
    return X


def _zscore(
    X: np.ndarray,
    axis: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Z-score along axis.
    axis=0 => standardize each feature (recommended default).
    axis=1 => row-wise standardization.
    """
    mu = np.nanmean(X, axis=axis, keepdims=True)
    sd = np.nanstd(X, axis=axis, keepdims=True)
    sd = np.maximum(sd, eps)
    return (X - mu) / sd


@dataclass
class IDFThresholdResult:
    variance_threshold: float     # e.g., 0.90, 0.95, 0.99, 0.9975
    intrinsic_dim: int            # k at that threshold
    idf: float                    # k / n_features
    cumvar_at_k: float            # cumulative variance at k (≈ threshold)
    success_prob: float           # P_success = 1 - idf
    complexity_at_thr: float      # cumvar_at_k / idf^2


@dataclass
class IDFAnalysisResult:
    dataset_name: str
    n_samples: int
    n_features: int

    # PCA curve
    explained_variance_ratio: List[float]
    cumulative_variance: List[float]

    # Threshold-wise details
    threshold_results: List[IDFThresholdResult]

    # Aggregates (dataset-level)
    mean_idf: float               # mean over thresholds
    auc_cumvar_vs_idf: float      # AUC of cumvar vs IDF (raw, not normalized)
    psi_star: float               # psi* = (AUC)^s with s=2
    loss: float                   # Loss(psi*) = (psi_star / AUC^s - 1)^2 (≈ 0)
    complexity_score: float       # mean over thresholds of [cumvar_at_k / idf^2]
    foe: float                    # FOE = 1 / (mean_idf^2) if mean_idf>0 else np.inf

    def to_dict(self) -> Dict[str, Any]:
        # dataclasses.asdict already handles nested dataclasses nicely
        return asdict(self)


class TabularIDFAnalyzer:
    """
    Intrinsic dimensionality + IDF + FOE + complexity + success prob, as in your theory.

    - IDF (per threshold):       idf = k / n_features
    - Complexity (per threshold): C_thr = cumvar_at_k / idf^2
    - Dataset complexity:         Complexity = mean_thr C_thr
    - AUC: area under curve (IDF, cumvar_at_k), raw trapezoidal rule
    - psi*:                        psi* = AUC^s, with fixed s=2
    - FOE:                         FOE = psi* / (AUC * mean_idf)^s = 1 / (mean_idf^2)
    - Success probability:        P_success = 1 - idf

    Notes:
      * s is fixed to 2 by default (matches your write-up).
      * AUC here is NOT normalized; it will be small for tiny IDF ranges,
        matching the tiny AUC numbers in your tables (e.g., 1e-3, 1e-2, ...).
    """

    def __init__(
        self,
        variance_thresholds: Iterable[float] = (0.90, 0.95, 0.99, 0.9975),
        scale: str = "zscore",                 # {"zscore", "none"}
        zscore_axis: int = 0,                  # 0=feature-wise, 1=row-wise
        impute: str = "mean",                  # {"mean", "median", "most_frequent", "none"}
        pca_solver: str = "full",
        random_state: Optional[int] = 0,
        s: float = 2.0,                        # sensitivity exponent (fixed at 2)
        # Optional overrides if you ever want to change behavior:
        success_prob_fn: Optional[Callable[[float, int, int, float], float]] = None,
    ):
        self.variance_thresholds = tuple(float(v) for v in variance_thresholds)
        self.scale = scale
        self.zscore_axis = int(zscore_axis)
        self.impute = impute
        self.pca_solver = pca_solver
        self.random_state = random_state
        self.s = float(s)

        # Success probability: default to your equation P_success = 1 - IDF
        self.success_prob_fn = success_prob_fn or self._default_success_prob

        # fitted artifacts
        self._pca: Optional[PCA] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None
        self._cumulative_variance: Optional[np.ndarray] = None

    # ---------------------------
    # Defaults
    # ---------------------------
    @staticmethod
    def _default_success_prob(idf: float, k: int, d: int, thr: float) -> float:
        """
        Default success probability:
            P_success = 1 - IDF = 1 - k/d
        Clamped into [0, 1].
        """
        return float(np.clip(1.0 - idf, 0.0, 1.0))

    # ---------------------------
    # Core pipeline
    # ---------------------------
    def _preprocess(self, X: ArrayLike) -> np.ndarray:
        Xn = _to_numpy(X).astype(float, copy=False)

        # Drop columns that are entirely NaN
        col_all_nan = np.all(~np.isfinite(Xn), axis=0)
        Xn = Xn[:, ~col_all_nan]

        # Impute
        if self.impute and self.impute.lower() != "none":
            imputer = SimpleImputer(strategy=self.impute)
            Xn = imputer.fit_transform(Xn)

        # Scale
        if self.scale.lower() == "zscore":
            Xn = _zscore(Xn, axis=self.zscore_axis)
        elif self.scale.lower() == "none":
            pass
        else:
            raise ValueError(f"Unknown scale='{self.scale}'. Use 'zscore' or 'none'.")

        # Replace any remaining nan/inf
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        return Xn

    def fit(self, X: ArrayLike) -> "TabularIDFAnalyzer":
        Xp = self._preprocess(X)

        self._pca = PCA(svd_solver=self.pca_solver, random_state=self.random_state)
        self._pca.fit(Xp)

        evr = np.asarray(self._pca.explained_variance_ratio_, dtype=float)
        cum = np.cumsum(evr)

        self._explained_variance_ratio = evr
        self._cumulative_variance = cum
        return self

    def _require_fit(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._explained_variance_ratio is None or self._cumulative_variance is None:
            raise RuntimeError("Call fit(X) before analyze().")
        return self._explained_variance_ratio, self._cumulative_variance

    @staticmethod
    def _intrinsic_dim_from_cumvar(cumvar: np.ndarray, threshold: float) -> int:
        """
        Smallest k such that cumvar[k-1] >= threshold.
        If never reaches threshold, use full rank.
        """
        idx = int(np.argmax(cumvar >= threshold))
        if cumvar[idx] < threshold:  # never reaches
            return int(len(cumvar))
        return idx + 1

    @staticmethod
    def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.trapz(y, x))

    def analyze(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,  # kept for API symmetry; unused
        dataset_name: str = "dataset",
        idf_denominator: str = "n_features",  # {"n_features"} now, sqrt option kept for experimentation
        normalize_auc: bool = False,          # IMPORTANT: default False to match your tiny AUC numbers
    ) -> IDFAnalysisResult:
        """
        Compute PCA, IDF curve, complexity, FOE, psi*, success probabilities, etc.

        idf definition:
          - default: idf = k / d  (k intrinsic dim, d number of features; matches paper)
          - option:  idf = k / sqrt(d) (only if you explicitly ask for it)
        """
        Xp = self._preprocess(X)
        n, d = Xp.shape

        # Always refit PCA per dataset
        self.fit(Xp)
        evr, cum = self._require_fit()

        thr_results: List[IDFThresholdResult] = []
        idfs: List[float] = []
        cum_points: List[float] = []
        complexity_per_thr: List[float] = []

        for thr in self.variance_thresholds:
            thr = float(thr)
            k = self._intrinsic_dim_from_cumvar(cum, thr)
            cum_at_k = float(cum[min(k - 1, len(cum) - 1)])

            if idf_denominator == "n_features":
                denom = float(d)
            elif idf_denominator == "sqrt_n_features":
                denom = float(np.sqrt(d))
            else:
                raise ValueError("idf_denominator must be 'n_features' or 'sqrt_n_features'.")

            idf = float(k / max(denom, 1.0))

            # Success probability from your Eq. (7)
            sp = float(self.success_prob_fn(idf, k, d, thr))

            # Complexity at this threshold: C_thr = cumvar_at_k / idf^2
            if idf > 0.0:
                c_thr = float(cum_at_k / (idf ** (2.0)))
            else:
                c_thr = float("inf")

            thr_results.append(
                IDFThresholdResult(
                    variance_threshold=thr,
                    intrinsic_dim=int(k),
                    idf=float(idf),
                    cumvar_at_k=float(cum_at_k),
                    success_prob=float(sp),
                    complexity_at_thr=float(c_thr),
                )
            )
            idfs.append(idf)
            cum_points.append(cum_at_k)
            complexity_per_thr.append(c_thr)

        idfs_arr = np.asarray(idfs, dtype=float)
        cum_points_arr = np.asarray(cum_points, dtype=float)

        # Sort by IDF for a stable AUC (IDF is the x-axis)
        order = np.argsort(idfs_arr)
        x = idfs_arr[order]
        y_curve = cum_points_arr[order]

        # AUC over IDF–cumvar curve (raw by default)
        auc = self._auc_trapz(x, y_curve)
        auc_raw = float(auc)

        if normalize_auc and np.isfinite(auc_raw) and len(x) >= 2:
            x_range = float(x[-1] - x[0])
            auc_norm = auc_raw / max(x_range, 1e-12)  # max area is x_range * 1.0
            auc_norm = float(np.clip(auc_norm, 0.0, 1.0))
            auc_for_loss = auc_norm
        else:
            auc_norm = auc_raw
            auc_for_loss = auc_raw

        mean_idf = float(np.nanmean(idfs_arr)) if len(idfs_arr) else float("nan")

        # psi* = AUC^s
        if np.isfinite(auc_for_loss) and auc_for_loss > 0.0:
            psi_star = float(auc_for_loss ** self.s)
            # Loss(psi) = (psi / AUC^s - 1)^2 at psi=psi_star -> ≈ 0
            loss = float((psi_star / (auc_for_loss ** self.s) - 1.0) ** 2)
        else:
            psi_star = float("nan")
            loss = float("nan")

        # Complexity score: mean over thresholds of C_thr = cumvar_at_k / idf^2
        if len(complexity_per_thr) > 0:
            complexity_score = float(np.nanmean(np.asarray(complexity_per_thr, dtype=float)))
        else:
            complexity_score = float("nan")

        # FOE from Eq. (3) with psi* and s=2:
        # FOE = psi_star / (AUC * mean_idf)^s = 1 / mean_idf^2 (when psi_star = AUC^s)
        if mean_idf > 0.0 and np.isfinite(mean_idf):
            foe = float(1.0 / (mean_idf ** self.s))
        else:
            foe = float("inf")  # degenerate case: IDF≈0 → FOE infinite

        return IDFAnalysisResult(
            dataset_name=str(dataset_name),
            n_samples=int(n),
            n_features=int(d),
            explained_variance_ratio=evr.tolist(),
            cumulative_variance=cum.tolist(),
            threshold_results=thr_results,
            mean_idf=mean_idf,
            auc_cumvar_vs_idf=auc_norm,
            psi_star=psi_star,
            loss=loss,
            complexity_score=complexity_score,
            foe=foe,
        )

    # ---------------------------
    # Plotting helpers
    # ---------------------------
    @staticmethod
    def plot_cumulative_variance_curve(
        result: IDFAnalysisResult,
        ax=None,
        title: Optional[str] = None,
        show_threshold_markers: bool = True,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        cum = np.asarray(result.cumulative_variance, dtype=float)
        ax.plot(np.arange(1, len(cum) + 1), cum, linewidth=2)

        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title(title or f"Cumulative Explained Variance — {result.dataset_name}")
        ax.grid(True)

        if show_threshold_markers:
            for tr in result.threshold_results:
                ax.scatter([tr.intrinsic_dim], [tr.cumvar_at_k], s=60, zorder=5)
                ax.annotate(
                    f"{tr.variance_threshold:.4g}",
                    (tr.intrinsic_dim, tr.cumvar_at_k),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                )
        return ax

    @staticmethod
    def plot_success_prob_vs_idf(
        result: IDFAnalysisResult,
        ax=None,
        title: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        x = [tr.idf for tr in result.threshold_results]
        y = [tr.success_prob for tr in result.threshold_results]

        order = np.argsort(np.asarray(x, dtype=float))
        xs = np.asarray(x, dtype=float)[order]
        ys = np.asarray(y, dtype=float)[order]

        ax.plot(xs, ys, linewidth=2)
        ax.scatter(xs, ys, s=70, zorder=5)

        ax.set_xlabel("Intrinsic Dimensionality Factor (IDF)")
        ax.set_ylabel("Success Probability")
        ax.set_title(title or f"Success Probability vs IDF — {result.dataset_name}")
        ax.grid(True)
        return ax

    @staticmethod
    def plot_cumvar_vs_idf(
        result: IDFAnalysisResult,
        ax=None,
        title: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        x = [tr.idf for tr in result.threshold_results]
        y = [tr.cumvar_at_k for tr in result.threshold_results]

        order = np.argsort(np.asarray(x, dtype=float))
        xs = np.asarray(x, dtype=float)[order]
        ys = np.asarray(y, dtype=float)[order]

        ax.plot(xs, ys, linewidth=2)
        ax.scatter(xs, ys, s=70, zorder=5)

        ax.set_xlabel("Intrinsic Dimensionality Factor (IDF)")
        ax.set_ylabel("Cumulative Variance Explained (at k)")
        ax.set_title(title or f"Cumulative Variance vs IDF — {result.dataset_name}")
        ax.grid(True)
        return ax

    @staticmethod
    def save_all_figures(
        result: IDFAnalysisResult,
        out_prefix: str,
        dpi: int = 300,
    ) -> Dict[str, str]:
        """
        Saves 3 figures:
          1) cumulative variance curve
          2) success prob vs idf
          3) cumulative variance vs idf

        Returns dict of {name: filepath}.
        """
        import matplotlib.pyplot as plt

        paths: Dict[str, str] = {}

        # 1) cumvar vs #components
        fig, ax = plt.subplots(figsize=(8, 5))
        TabularIDFAnalyzer.plot_cumulative_variance_curve(result, ax=ax)
        p1 = f"{out_prefix}_cumvar_curve.png"
        fig.tight_layout()
        fig.savefig(p1, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths["cumvar_curve"] = p1

        # 2) success prob vs IDF
        fig, ax = plt.subplots(figsize=(7, 5))
        TabularIDFAnalyzer.plot_success_prob_vs_idf(result, ax=ax)
        p2 = f"{out_prefix}_successprob_vs_idf.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths["successprob_vs_idf"] = p2

        # 3) cumvar vs IDF
        fig, ax = plt.subplots(figsize=(7, 5))
        TabularIDFAnalyzer.plot_cumvar_vs_idf(result, ax=ax)
        p3 = f"{out_prefix}_cumvar_vs_idf.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths["cumvar_vs_idf"] = p3

        return paths


# ---------------------------
# Convenience: batch runner
# ---------------------------
def analyze_many(
    datasets: Dict[str, ArrayLike],
    analyzer: Optional[TabularIDFAnalyzer] = None,
    y_map: Optional[Dict[str, ArrayLike]] = None,
    **analyze_kwargs,
) -> Dict[str, IDFAnalysisResult]:
    """
    datasets: {"name": X, ...}
    y_map: optional {"name": y, ...} (y is ignored by analysis, but kept for symmetry).
    """
    analyzer = analyzer or TabularIDFAnalyzer()
    out: Dict[str, IDFAnalysisResult] = {}
    for name, X in datasets.items():
        y = None if y_map is None else y_map.get(name, None)
        out[name] = analyzer.analyze(X=X, y=y, dataset_name=name, **analyze_kwargs)
    return out