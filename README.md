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

DynaTab is a neuro-inspired tabular deep learning model for high-dimensional tabular data that tackles the Column Permutation Problem by dynamically reordering features instead of treating them as a fixed set. It predicts when feature ordering is beneficial using an intrinsic-dimensionality-based IDF/FOE criterion, then applies dynamic feature ordering (DFO) to rewire feature graphs and produce a task-aware global sequence. This reordered input is processed by an order-aware fusion block combining positional embeddings (OPE), importance gating (PIGL), and dynamic masked attention (DMA) on top of a sequential backbone (Transformer, DAE, LSTM, Mamba, or DAE-MHA-LSTM). It also empirically group tabular datasets into 5 categories. Across 36 real-world datasets and over 45 baselines, DynaTab achieves strong, statistically significant gains, particularly in high-dimensional low-sample-size (HDLSS) and other complex regimes, positioning dynamic feature ordering as a powerful paradigm for tabular deep learning for high-dimensional tabular data.


## Citation

Al Zadid Sultan Bin Habib, Gianfranco Doretto, and Donald A. Adjeroh.  
“DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data.”  
In *AAAI 2026 First International Workshop on Neuro for AI \& AI for Neuro: Towards Multi-Modal Natural Intelligence (NeuroAI) Workshop Proceedings (PMLR)*, 2026.

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
