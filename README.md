# DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Neuroplasticity](https://img.shields.io/badge/Neuroplasticity-Dynamic%20Feature%20Ordering-blueviolet)
![OPE](https://img.shields.io/badge/OPE-Order--Aware%20Positional%20Embedding-orange)
![PIGL](https://img.shields.io/badge/PIGL-Positional%20Importance%20Gating%20Layer-blueviolet)
![DMA](https://img.shields.io/badge/DMA-Dynamic%20Masked%20Attention-orange)
![Backbone](https://img.shields.io/badge/Backbone-DAE%2FLSTM%2FDAE--MHA--LSTM%2FTransformer%2FMamba-informational)
![IDF Analyzer](https://img.shields.io/badge/IDF%20Analyzer-Feature%20Ordering%20When%20to%20Use%3F-success)
![Model](https://img.shields.io/badge/Model-DynaTab-skyblue)
![Conference](https://img.shields.io/badge/Conference-AAAI%202026%20NeuroAI%20Workshop-blue)
[![Citation](https://img.shields.io/badge/Cite%20Us-PMLR--AAAI--2026--NeuroAI--Workshop-red)](https://neuroai-multimodal-workshop.github.io/)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

<p align="center">
  <img src="DynaTab_Architecture.jpg" alt="DynaTab Architecture" width="900">
</p>

DynaTab is a neuro-inspired tabular deep learning model for high-dimensional tabular data that tackles the Column Permutation Problem by dynamically reordering features instead of treating them as a fixed set. It predicts when feature ordering is beneficial using an intrinsic-dimensionality-based IDF/FOE criterion, then applies dynamic feature ordering (DFO) to rewire feature graphs and produce a task-aware global sequence. This reordered input is processed by an order-aware fusion block combining positional embeddings (OPE), importance gating (PIGL), and dynamic masked attention (DMA) on top of a sequential backbone (Transformer, DAE, LSTM, Mamba, or DAE-MHA-LSTM). It also empirically group tabular datasets into 5 categories. Across 36 real-world datasets and over 45 baselines, DynaTab achieves strong, statistically significant gains, particularly in high-dimensional low-sample-size (HDLSS) and other complex regimes, positioning dynamic feature ordering as a powerful paradigm for order-sensitive backbones in tabular deep learning for high-dimensional tabular data.

## Citation

Al Zadid Sultan Bin Habib, Gianfranco Doretto, and Donald A. Adjeroh.   “DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data.”   In *AAAI 2026 First International Workshop on Neuro for AI \& AI for Neuro: Towards Multi-Modal Natural Intelligence (NeuroAI) Workshop Proceedings (PMLR)*, 2026.

Bibtex:
```bash
@inproceedings{habib2026dynatab,
  title     = {{DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data}},
  author    = {Habib, Al Zadid Sultan Bin and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the AAAI 2026 First International Workshop on Neuro for AI \& AI for Neuro: Towards Multi-Modal Natural Intelligence (NeuroAI)},
  year      = {2026},
  series    = {PMLR}
}
```

## Files and Repository Structure

### Python package: `dynatab/`

This folder contains the core DynaTab implementation (15 Python modules):

- `__init__.py` - Package initializer and high-level API exports.
- `model.py` - Main DynaTab model definition and wiring of all sub-modules.
- `dfo.py` - Dynamic Feature Ordering (DFO) module and clustering/graph construction.
- `ope.py` - Order-Aware Positional Embedding (OPE) implementation.
- `pigl.py` - Positional Importance Gating Layer (PIGL).
- `dma.py` - Dynamic Masked Attention (DMA) block.
- `seqprobinary.py` - Training loop / utilities for **binary classification**.
- `seqpromulti.py` - Training loop / utilities for **multiclass classification**.
- `seqproregression.py` - Training loop / utilities for **regression**.
- `preprocess.py` - Data preprocessing and tabular input utilities (splits, scaling, etc.).
- `metrics.py` - Evaluation metrics and helper functions.
- `estimator.py` - High-level estimator wrapper for running experiments (sklearn-style API).
- `idf_analyzer.py` - Intrinsic Dimensionality Factor (IDF) + FOE analyzer: “Feature Ordering – When to Use?”.
- `customloss.py` - Custom loss functions used by DynaTab.
- `trainer.py` - Generic training / validation loop utilities shared across tasks.

### Notebooks

- **`DynaTab Dataset Complexity Analysis.ipynb`**  
  Contains the experiments for the **“Feature Ordering – When to Use?”** section, including IDF / FOE computation across datasets.

- **`DynaTab IDF Analyzer.ipynb`**  
  Shows how to install/import the `dynatab` package and use `TabularIDFAnalyzer` to compute dataset complexity metrics with demo runs.  
  The code cells illustrate how to use DynaTab to assess when feature ordering is appropriate for a given dataset.

- **`DynaTab_Experiment1.ipynb`**  
  Demonstrates how to use DynaTab for binary classification, multiclass classification, and regression, with or without Optuna-based hyperparameter tuning.

- **`DynaTab_Experiment2.ipynb`**  
  Demonstrates DynaTab on the GLI-85 HDLSS dataset for binary classification, without Optuna tuning, using Mamba or LSTM as the sequential processor backbone.
- *N.B.: Demo runs only contain less number of epochs or Optuna trials. For complete run, please use proper number of Optuna trials to search and find optimum hyperparameters.*

### Other top-level files

- **`requirements.txt`** - Python dependencies required to run the DynaTab package and notebooks.
- **`DynaTab_Architecture.jpg`** - High-level architecture diagram of the DynaTab framework.
- **`LICENSE`** - MIT license for this repository.
- **`README.md`** - Project overview, usage instructions, and citation information.
- **`.gitignore`** - Standard Git ignore rules for Python and Jupyter projects.


### Tested Environment

- Python 3.8+
- torch 2.5.1+cu121 (CUDA 12.1)
- numpy 1.26.4
- pandas 2.2.3
- scikit-learn 1.5.2
- matplotlib 3.10.0
- scipy 1.11.4
- kmeans_gpu 0.0.5

### Recommended PyTorch install (GPU, CUDA 12.1)

```bash
pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
```

## Installation

You can install **DynaTab** in several ways depending on your workflow.

---

### Option 1: Clone the Repository (Recommended for Development)

```bash
git clone https://github.com/zadid6pretam/DynaTab.git
cd DynaTab
pip install -r requirements.txt
pip install -e .
```

### Option 2: Install Directly from GitHub (No Cloning Needed)

```bash
pip install "git+https://github.com/zadid6pretam/DynaTab.git"
```

### Option 3: Use a Virtual Environment

```bash
python -m venv dynatab-env
source dynatab-env/bin/activate  # On Windows: dynatab-env\Scripts\activate

git clone https://github.com/zadid6pretam/DynaTab.git
cd DynaTab
pip install -r requirements.txt
pip install -e .
```

### Option 4: Local Install Without Editable Mode

```bash
git clone https://github.com/zadid6pretam/DynaTab.git
cd DynaTab
pip install -r requirements.txt
pip install .
```

### Option 5: Install from PyPI (Planned)

```bash
pip install dynatab
```

## Example Usage

Below are minimal examples for using **DynaTab** on standard binary, multiclass, and regression tasks.  
For full HDLSS experiments and Optuna sweeps, see the accompanying Jupyter notebooks.

---

### 1. Binary Classification (Breast Cancer)

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from dynatab import (
    DynaTabClassifier,
    DFOConfig,
    TrainConfig,
    LossConfig,
)

# -----------------------------
# Data
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0/1 labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# -----------------------------
# DynaTab configs
# -----------------------------
dfo_cfg = DFOConfig(
    metric="manhattan",
    num_clusters=2,
    order="ascending",
    mutation_prob=0.0,
    tolerance=1e-3,
    seed=42,
)

train_cfg = TrainConfig(
    epochs=100,
    lr=1e-3,
    batch_size=256,
    print_every=20,
)

loss_cfg = LossConfig(
    loss_mode="DFO",      # "standard" | "dispersion" | "DFO"
    lambda_disp=0.0,
    lambda_global=0.0,
)

# -----------------------------
# Model: DynaTabClassifier
# -----------------------------
clf = DynaTabClassifier(
    task="binary",
    backbone="Transformer",   # or "LSTM", "DAE", "Mamba", ...
    embedding_dim=128,
    dfo_cfg=dfo_cfg,
    train_cfg=train_cfg,
    loss_cfg=loss_cfg,
    eval_metrics=["acc"],
    device=None,              # auto-selects CUDA/CPU
    standardize=True,         # train-only impute + standardize
)

clf.fit(X_train, y_train)
metrics = clf.score(X_test, y_test, metrics=["acc"])
print(f"Test Accuracy (Breast Cancer): {metrics['acc']:.4f}")
```

### 2. Multiclass Classification (Iris)

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from dynatab import (
    DynaTabClassifier,
    DFOConfig,
    TrainConfig,
    LossConfig,
)

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 3 classes: 0,1,2

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

dfo_cfg = DFOConfig(
    metric="variance",
    num_clusters=3,
    order="descending",
    mutation_prob=0.1,
    tolerance=1e-3,
    seed=42,
)

train_cfg = TrainConfig(
    epochs=80,
    lr=1e-3,
    batch_size=64,
    print_every=20,
)

loss_cfg = LossConfig(
    loss_mode="standard",
    lambda_disp=0.0,
    lambda_global=0.0,
)

clf = DynaTabClassifier(
    task="multiclass",
    num_classes=3,
    backbone="Transformer",
    embedding_dim=64,
    dfo_cfg=dfo_cfg,
    train_cfg=train_cfg,
    loss_cfg=loss_cfg,
    eval_metrics=["acc"],
    device=None,
    standardize=True,
)

clf.fit(X_train, y_train)
metrics = clf.score(X_test, y_test, metrics=["acc"])
print(f"Test Accuracy (Iris): {metrics['acc']:.4f}")
```

### 3. Regression (Diabetes)

```python
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from dynatab import (
    DynaTabRegressor,
    DFOConfig,
    TrainConfig,
    LossConfig,
)

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

dfo_cfg = DFOConfig(
    metric="correlation",
    num_clusters=3,
    order="ascending",
    mutation_prob=0.1,
    tolerance=1e-3,
    seed=42,
)

train_cfg = TrainConfig(
    epochs=120,
    lr=1e-3,
    batch_size=128,
    print_every=20,
)

loss_cfg = LossConfig(
    loss_mode="standard",   # for regression we typically keep it standard
    lambda_disp=0.0,
    lambda_global=0.0,
)

reg = DynaTabRegressor(
    backbone="Transformer",
    embedding_dim=64,
    dfo_cfg=dfo_cfg,
    train_cfg=train_cfg,
    loss_cfg=loss_cfg,
    eval_metrics=["r2"],    # e.g., R^2
    device=None,
    standardize=True,
)

reg.fit(X_train, y_train)
metrics = reg.score(X_test, y_test, metrics=["r2"])
print(f"Test R² (Diabetes): {metrics['r2']:.4f}")
```

### 4. Advanced: 5-Fold CV + Optuna Hyperparameter Tuning

For full HDLSS experiments, repeated CV, and Optuna-based tuning (Transformer, LSTM, DAE, Mamba backbones) on real datasets such as **AI-d_case5**, **ADNI_AD123**, **GLI-85**, and others, see:

- **`DynaTab_Experiment1.ipynb`** – Binary & multiclass classification and regression (with / without Optuna-based hyperparameter tuning).
- **`DynaTab_Experiment2.ipynb`** – HDLSS case studies (e.g., GLI-85 with Mamba/LSTM backbones).
- **`DynaTab Dataset Complexity Analysis.ipynb`** and **`DynaTab IDF Analyzer.ipynb`** – Intrinsic dimensionality and “when to use feature ordering” analysis.
- You can tweak the metrics / epochs / DFO settings if you want them lighter or closer to the paper defaults.

## Previous Work: TabSeq

DynaTab builds on our earlier work on feature ordering for tabular data:

- **TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering**  
  GitHub: https://github.com/zadid6pretam/TabSeq  
  Springer (ICPR 2024 proceedings): https://link.springer.com/chapter/10.1007/978-3-031-78128-5_27  

If you are interested in:
- MHA-DAE-guided sequential tabular models,
- Cluster-guided feature ordering, and  
- Baseline comparison to classical ML and other deep models,

please also refer to the **TabSeq** repository and its accompanying paper as the foundational precursor to DynaTab.
