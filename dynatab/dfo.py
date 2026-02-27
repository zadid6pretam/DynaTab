# This part of the code implements Dynamic Feature Ordering (DFO)
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import torch

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import mutual_info_score

try:
    # Optional: GPU KMeans backend
    from kmeans_gpu import KMeans as TorchKMeansGPU
    _HAS_KMEANS_GPU = True
except Exception:
    TorchKMeansGPU = None
    _HAS_KMEANS_GPU = False

try:
    # Optional CPU fallback
    from sklearn.cluster import KMeans as SklearnKMeans
    _HAS_SKLEARN_KMEANS = True
except Exception:
    SklearnKMeans = None
    _HAS_SKLEARN_KMEANS = False


MetricName = Literal[
    "variance",
    "euclidean",
    "correlation",
    "manhattan",
    "kl_divergence",
    "mutual_info",
    "js_divergence",
    "wasserstein_distance",
]


@dataclass
class DFOConfig:
    metric: MetricName = "variance"
    num_clusters: int = 5
    order: Literal["ascending", "descending"] = "ascending"
    mutation_prob: float = 0.0
    tolerance: float = 0.01
    seed: int = 42


class DynamicFeatureOrdering:
    """
    Dynamic Feature Ordering (DFO) module.

    Key methods:
      - reorder_and_evaluate(X_train, y_train=None) -> (X_reordered, centroids, labels, global_order)

    Notes:
      - For library hygiene, we do NOT reinitialize RMM/CuPy pools at import-time.
        If you want RMM pooling, call `enable_rmm_pool()` explicitly from your script.
    """

    def __init__(self, config: Optional[DFOConfig] = None, device: Optional[Union[str, torch.device]] = None):
        self.config = config or DFOConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # best-effort reproducibility
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

    # ---------- Optional memory pool setup ----------
    def enable_rmm_pool(
        self,
        pool_allocator: bool = True,
        managed_memory: bool = False,
        initial_pool_size: int = 0,
    ) -> None:
        """
        Optional: enable RMM pool allocator for CuPy allocations.
        Call this ONCE early in your program if you need it.

        Safe to skip entirely if you don't use CuPy directly.
        """
        try:
            import rmm  # type: ignore
            torch.cuda.empty_cache()
            rmm.reinitialize(
                pool_allocator=pool_allocator,
                managed_memory=managed_memory,
                initial_pool_size=initial_pool_size,
            )
        except Exception:
            # If rmm isn't available, just ignore.
            pass

    # ---------- GPU / CPU KMeans ----------
    def gpu_kmeans(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, torch.Tensor]:
        """
        wrapper:
          - labels: np.ndarray (N,)
          - centers: torch.Tensor (K, D) on self.device
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features). Got shape={X.shape}")

        # Prefer GPU backend if available and device is cuda
        if _HAS_KMEANS_GPU and self.device.type == "cuda":
            X_tensor = torch.tensor(X, device=self.device, dtype=torch.float32)

            kmeans_gpu = TorchKMeansGPU(n_clusters=n_clusters)

            # kmeans_gpu expects batch dim + dummy mask
            dummy = torch.ones(1, 1, X_tensor.shape[0], device=self.device, dtype=torch.float32)
            centroids, _ = kmeans_gpu(X_tensor.unsqueeze(0), dummy)  # (1, K, D)
            centers = centroids[0].to(dtype=torch.float32)          # (K, D) on device

            # assign labels
            dists = torch.cdist(X_tensor, centers)                  # (N, K)
            labels = torch.argmin(dists, dim=1).detach().cpu().numpy().astype(np.int64)
            return labels, centers

        # CPU fallback (sklearn)
        if not _HAS_SKLEARN_KMEANS:
            raise RuntimeError("No GPU KMeans available and sklearn KMeans is not installed.")

        km = SklearnKMeans(n_clusters=n_clusters, random_state=self.config.seed, n_init="auto")
        labels = km.fit_predict(X).astype(np.int64)
        centers = torch.tensor(km.cluster_centers_, device=self.device, dtype=torch.float32)
        return labels, centers

    # ---------- Metrics ----------
    @staticmethod
    def _kl_divergence_torch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        return torch.sum(p * torch.log(p / (q + 1e-10)))

    @staticmethod
    def _calculate_cpu_metric(p: pd.Series, q: pd.Series, metric: MetricName) -> float:
        if metric == "mutual_info":
            return float(mutual_info_score(p, q))
        elif metric == "js_divergence":
            p2, q2 = p.align(q, fill_value=1e-10)
            return float(jensenshannon(p2, q2))
        elif metric == "wasserstein_distance":
            p2, q2 = p.align(q, fill_value=1e-10)
            return float(wasserstein_distance(p2, q2))
        else:
            raise ValueError(f"Unsupported CPU metric '{metric}'.")

    def calculate_metric_matrix(self, X: np.ndarray, metric: MetricName) -> torch.Tensor:
        """
        Returns: (D, D) torch.Tensor on self.device (float32)
        """
        X_tensor = torch.tensor(X, device=self.device, dtype=torch.float32)
        D = X_tensor.shape[1]

        if metric == "variance":
            # your original expression; keep behavior
            metric_matrix = torch.var(X_tensor.unsqueeze(2) - X_tensor.unsqueeze(1), dim=0).abs()

        elif metric == "euclidean":
            metric_matrix = torch.cdist(X_tensor.T, X_tensor.T, p=2)

        elif metric == "correlation":
            X_norm = (X_tensor - X_tensor.mean(dim=0)) / (X_tensor.std(dim=0) + 1e-10)
            metric_matrix = torch.abs(torch.mm(X_norm.T, X_norm) / X_tensor.size(0))

        elif metric == "manhattan":
            metric_matrix = torch.cdist(X_tensor.T, X_tensor.T, p=1)

        elif metric == "kl_divergence":
            metric_matrix = torch.zeros((D, D), device=self.device, dtype=torch.float32)
            # NOTE: O(D^2) loops. Consider batching later if needed.
            for i in range(D):
                for j in range(i + 1, D):
                    value = self._kl_divergence_torch(X_tensor[:, i], X_tensor[:, j])
                    metric_matrix[i, j] = metric_matrix[j, i] = value

        else:
            # CPU-only distributional metrics via value_counts (as you had)
            X_df = pd.DataFrame(X)
            mm = np.zeros((D, D), dtype=np.float32)
            for i in range(D):
                for j in range(i + 1, D):
                    p = X_df.iloc[:, i].value_counts(normalize=True).sort_index()
                    q = X_df.iloc[:, j].value_counts(normalize=True).sort_index()
                    value = self._calculate_cpu_metric(p, q, metric)
                    mm[i, j] = mm[j, i] = value
            metric_matrix = torch.tensor(mm, device=self.device, dtype=torch.float32)

        return metric_matrix.to(dtype=torch.float32)

    # ---------- Ordering / rewiring ----------
    def minimize_dispersion(self, adj_matrix: torch.Tensor, order: str) -> np.ndarray:
        weight_sum = torch.sum(adj_matrix, dim=1)
        sorted_idx = torch.argsort(weight_sum, descending=(order != "ascending"))
        return sorted_idx.detach().cpu().numpy()

    def gpu_rewire_connections(
        self,
        adj_matrix: torch.Tensor,
        original_ordering: np.ndarray,
        mutation_prob: float,
        tolerance: float,
        order: str,
    ) -> np.ndarray:
        """
        Mutation / rewiring step. Returns either original_ordering or a new ordering.
        """
        if torch.rand(1, device=self.device).item() >= mutation_prob:
            return original_ordering

        prev = adj_matrix.clone()

        # 25% quantile threshold via kthvalue on GPU/CPU torch
        flat = adj_matrix.reshape(-1)
        k = max(1, int(flat.numel() * 0.25))
        threshold, _ = flat.kthvalue(k, dim=0)

        adj2 = torch.where(adj_matrix < threshold, torch.zeros_like(adj_matrix), adj_matrix)

        # reassign any zero-row to top-central nodes
        centrality = adj2.sum(dim=1)
        high = torch.argsort(centrality, descending=True)

        for i in range(adj2.shape[0]):
            if adj2[i].sum() == 0:
                tgt = high[0] if high[0].item() != i else high[1]
                adj2[i, tgt] = centrality[tgt] * torch.rand(1, device=self.device)

        change = torch.abs(adj2 - prev).mean().item()
        if change < tolerance:
            return original_ordering

        return self.minimize_dispersion(adj2, order=order)

    @staticmethod
    def integrate_orderings(local_orderings: List[np.ndarray], cluster_distances: torch.Tensor) -> List[int]:
        feature_scores: dict[int, float] = {}
        for idx, ordering in enumerate(local_orderings):
            weight = 1.0 / (cluster_distances[idx].sum().item() + 1e-10)
            for pos, feat in enumerate(ordering.tolist()):
                feature_scores[feat] = feature_scores.get(feat, 0.0) + float(pos) * weight
        return sorted(feature_scores, key=feature_scores.get)

    # ---------- Main API ----------
    def reorder_and_evaluate(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train=None,  # kept for signature compatibility; unused here
        metric: Optional[MetricName] = None,
        num_clusters: Optional[int] = None,
        order: Optional[Literal["ascending", "descending"]] = None,
        mutation_prob: Optional[float] = None,
        tolerance: Optional[float] = None,
    ):
        """
        Returns:
          - X_reordered: DataFrame if input is DataFrame else np.ndarray
          - centroids: torch.Tensor (K, D) on device
          - labels: np.ndarray (N,)
          - global_order: List[int] indices
        """
        metric = metric or self.config.metric
        num_clusters = int(num_clusters or self.config.num_clusters)
        order = order or self.config.order
        mutation_prob = float(mutation_prob if mutation_prob is not None else self.config.mutation_prob)
        tolerance = float(tolerance if tolerance is not None else self.config.tolerance)

        if isinstance(X_train, pd.DataFrame):
            X_np = X_train.values
            colnames = list(X_train.columns)
        else:
            X_np = np.asarray(X_train)
            colnames = None

        labels, centroids = self.gpu_kmeans(X_np, n_clusters=num_clusters)
        cluster_distances = torch.cdist(centroids, centroids)

        local_orderings: List[np.ndarray] = []
        for i in range(num_clusters):
            data_i = X_np[labels == i]
            if data_i.shape[0] == 0:
                continue  # empty cluster; skip
            adj = self.calculate_metric_matrix(data_i, metric=metric)
            local = self.minimize_dispersion(adj, order=order)
            mutated = self.gpu_rewire_connections(adj, local, mutation_prob, tolerance, order=order)
            local_orderings.append(mutated)

        global_order = self.integrate_orderings(local_orderings, cluster_distances)

        if colnames is not None:
            cols = [colnames[i] for i in global_order]
            X_reordered = X_train[cols]
        else:
            X_reordered = X_np[:, global_order]

        return X_reordered, centroids, labels, global_order

# ==========================
# Public functional API
# ==========================
def reorder_and_evaluate(
    X_train,
    y_train=None,
    *,
    metric: Optional[MetricName] = None,
    num_clusters: Optional[int] = None,
    order: Optional[Literal["ascending", "descending"]] = None,
    mutation_prob: Optional[float] = None,
    tolerance: Optional[float] = None,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Convenience wrapper so users can call:
      from dynatab.dfo import reorder_and_evaluate
    without manually instantiating DynamicFeatureOrdering.
    """
    cfg = DFOConfig(
        metric=metric or DFOConfig().metric,
        num_clusters=int(num_clusters or DFOConfig().num_clusters),
        order=order or DFOConfig().order,
        mutation_prob=float(mutation_prob if mutation_prob is not None else DFOConfig().mutation_prob),
        tolerance=float(tolerance if tolerance is not None else DFOConfig().tolerance),
        seed=int(seed or DFOConfig().seed),
    )
    dfo = DynamicFeatureOrdering(config=cfg, device=device)
    return dfo.reorder_and_evaluate(
        X_train,
        y_train=y_train,
        metric=cfg.metric,
        num_clusters=cfg.num_clusters,
        order=cfg.order,
        mutation_prob=cfg.mutation_prob,
        tolerance=cfg.tolerance,
    )


def run_dfo(*args, **kwargs):
    """
    Stable alias for reorder_and_evaluate (API sugar).
    """
    return reorder_and_evaluate(*args, **kwargs)