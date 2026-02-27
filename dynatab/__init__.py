from __future__ import annotations

# --------------------
# Public API
# --------------------
__all__ = [
    # models
    "DynaTabBinary",
    "DynaTabMulti",
    "DynaTabRegression",

    # sklearn-like estimators
    "DynaTabClassifier",
    "DynaTabRegressor",

    # (optional) preprocessing helpers
    "Standardizer",
    "sanitize_numeric_dataframe",
    "impute_median_train",
    "fit_standardizer",
    "train_transform_preprocess",

    # building blocks
    "OrderAwarePositionalEmbedding",
    "PositionalImportanceGatingLayer",
    "PIGL",  # alias
    "DynamicMaskedAttention",
    "create_dma_mask",

    # DFO (feature ordering)
    "DFOConfig",
    "run_dfo",
    "reorder_and_evaluate",

    # training + config
    "TrainConfig",
    "ModelConfig",
    "LossConfig",
    "train_one_split",
    "evaluate_split",
    "cross_validate",
    "fit_from_dataframes_one_split",

    # optional: losses/metrics
    "CustomFeatureLoss",
    "evaluate_predictions",

    # IDF / FOE analyzer
    "TabularIDFAnalyzer",
    "analyze_many",
]

# --------------------
# Public models
# --------------------
from .model import DynaTabBinary, DynaTabMulti, DynaTabRegression

# --------------------
# Core building blocks
# --------------------
from .ope import OrderAwarePositionalEmbedding
from .pigl import PositionalImportanceGatingLayer, PIGL
from .dma import DynamicMaskedAttention, create_dma_mask

# --------------------
# Preprocess exports (safe import) (NEW)
# --------------------
try:
    from .preprocess import (
        Standardizer,
        sanitize_numeric_dataframe,
        impute_median_train,
        fit_standardizer,
        train_transform_preprocess,
    )
except Exception as e:
    _PREPROCESS_IMPORT_ERROR = e

    class Standardizer:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"dynatab.preprocess failed to import: {_PREPROCESS_IMPORT_ERROR}")

    def sanitize_numeric_dataframe(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.preprocess failed to import: {_PREPROCESS_IMPORT_ERROR}")

    def impute_median_train(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.preprocess failed to import: {_PREPROCESS_IMPORT_ERROR}")

    def fit_standardizer(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.preprocess failed to import: {_PREPROCESS_IMPORT_ERROR}")

    def train_transform_preprocess(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.preprocess failed to import: {_PREPROCESS_IMPORT_ERROR}")

# --------------------
# Estimator exports (safe import) (NEW)
# --------------------
try:
    from .estimator import DynaTabClassifier, DynaTabRegressor
except Exception as e:
    _ESTIMATORS_IMPORT_ERROR = e

    class DynaTabClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"dynatab.estimators failed to import: {_ESTIMATORS_IMPORT_ERROR}")

    class DynaTabRegressor:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"dynatab.estimators failed to import: {_ESTIMATORS_IMPORT_ERROR}")

# --------------------
# DFO exports (safe import)
# --------------------
try:
    from .dfo import DFOConfig, run_dfo, reorder_and_evaluate
except Exception as e:  # keep dynatab importable even if dfo.py is mid-edit
    _DFO_IMPORT_ERROR = e

    class DFOConfig:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"dynatab.dfo failed to import: {_DFO_IMPORT_ERROR}")

    def run_dfo(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.dfo failed to import: {_DFO_IMPORT_ERROR}")

    def reorder_and_evaluate(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.dfo failed to import: {_DFO_IMPORT_ERROR}")


# --------------------
# IDF analyzer exports (safe import)
# --------------------
try:
    from .idf_analyzer import TabularIDFAnalyzer, analyze_many
except Exception as e:  # keep dynatab importable even if idf_analyzer.py is mid-edit
    _IDF_IMPORT_ERROR = e

    class TabularIDFAnalyzer:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"dynatab.idf_analyzer failed to import: {_IDF_IMPORT_ERROR}")

    def analyze_many(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.idf_analyzer failed to import: {_IDF_IMPORT_ERROR}")

# --------------------
# Training utilities (safe import)
# --------------------
try:
    from .trainer import (
        TrainConfig,
        ModelConfig,
        LossConfig,
        train_one_split,
        evaluate_split,
        cross_validate,
        fit_from_dataframes_one_split,
    )
except Exception as e:  # keep dynatab importable even if trainer.py is mid-edit
    _TRAIN_IMPORT_ERROR = e

    class TrainConfig:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"dynatab.trainer failed to import: {_TRAIN_IMPORT_ERROR}")

    class ModelConfig:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"dynatab.trainer failed to import: {_TRAIN_IMPORT_ERROR}")

    class LossConfig:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"dynatab.trainer failed to import: {_TRAIN_IMPORT_ERROR}")

    def train_one_split(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.trainer failed to import: {_TRAIN_IMPORT_ERROR}")

    def evaluate_split(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.trainer failed to import: {_TRAIN_IMPORT_ERROR}")

    def cross_validate(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.trainer failed to import: {_TRAIN_IMPORT_ERROR}")

    def fit_from_dataframes_one_split(*args, **kwargs):  # type: ignore
        raise ImportError(f"dynatab.trainer failed to import: {_TRAIN_IMPORT_ERROR}")

# --------------------
# Loss + metrics
# --------------------
from .customloss import CustomFeatureLoss
from .metrics import evaluate_predictions