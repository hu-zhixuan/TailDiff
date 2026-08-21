# TailDiff: A Lightweight Conditional Diffusion Modeling Approach for Financial Tail-Risk Early Warning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Elsevier%20Standard%20Format-red.svg)](./表格与图片/paper.tex)

Official implementation and open-source benchmark reproduction suite for the paper:  
**"TailDiff: A Lightweight Conditional Diffusion Modeling Approach for Financial Tail-Risk Early Warning"**

---

## 🌟 Key Highlights

- **Lightweight 1D Causal Convolutional Backbone ($0.91$M Parameters):** Replaces heavy multi-head attention with 1D dilated residual convolutions ($RF=37 \ge 32$) modulated by Feature-wise Linear Modulation (FiLM) with realized volatility $RV_{20}$.
- **20-Step Deterministic DDIM Sampler ($5.09$ ms/sample):** Replaces 1000-step stochastic sampling with an ultra-fast ODE reverse trajectory, achieving millisecond-level CPU inference for live trading risk management.
- **Bilateral DCR Sweet-Spot Quality Gating ($[Q_{5\%}, Q_{95\%}]$):** Non-parametric Leave-One-Out (LOO) distance filter that adaptively suppresses training sample memorization and unphysical hallucinations while emerging $\lambda_{\text{eff}} \approx 15.20\times \sim 17.27\times$.
- **Strict Zero-Leakage Purged \& Embargoed Walk-Forward Cross-Validation:** Multi-fold rolling time-series framework with label purging and serial-correlation embargo buffers ($100\%$ pure out-of-sample evaluation).
- **Substantial Empirical Performance Gains:**
  - **Minority Tail Recall:** $+430\%$ to $+2000\%$ improvement (up to $\mathbf{51.13\%}$ on S\&P 500 and $\mathbf{31.02\%}$ on CSI 300).
  - **PR-AUC:** Doubled on S\&P 500 ($+102.7\%$, $0.1296 \to \mathbf{0.2626}$).
  - **Statistical Significance:** Confirmed via Wilcoxon signed-rank test across 35 Purged folds ($p = 0.0039 < 0.01$ ***).
  - **Practical Hedging Utility:** Cuts maximum portfolio drawdown from $-46.5\%$ to $-14.2\%$.

---

## 🏗️ System Architecture

![TailDiff Architecture](./表格与图片/figures/Fig1_TailDiff_Architecture.png)

TailDiff operates under a **Two-Stage Decoupled Paradigm**:
1. **Offline Generative Simulation:** Historical market data $\to$ EVT $\text{DaR}_{97.5\%}$ extreme labeling $\to$ 1D FiLM Conditional Diffusion $\to$ 20-Step DDIM ODE Sampling $\to$ DCR Bilateral Quality Gating.
2. **Online Real-Time Inference:** Fast tree-based classifiers (XGBoost / LightGBM / MLP) trained on the augmented manifold execute sub-millisecond risk warning and dynamic economic hedging.

---

## 📊 Benchmark Results (35-Fold Purged Walk-Forward CV)

| Market | Downstream Model | Augmentation Strategy | Tail Recall ($\uparrow$) | Precision ($\uparrow$) | F1-Score ($\uparrow$) | PR-AUC ($\uparrow$) | FPR ($\downarrow$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CSI 300** | **XGBoost** | No-Aug (Original) | 0.0093 | 0.0512 | 0.0152 | 0.1396 | **0.0116** |
| **CSI 300** | **XGBoost** | Replicate Baseline | 0.0139 | 0.0580 | 0.0238 | 0.1412 | **0.0097** |
| **CSI 300** | **XGBoost** | **TailDiff (Proposed, Ours)** | **0.2024** | **0.1860** | **0.1361** | **0.1773** | 0.1941 |
| **CSI 300** | **LightGBM** | **TailDiff (Proposed, Ours)** | **0.2097** | **0.1940** | **0.1379** | **0.1768** | 0.1986 |
| **CSI 300** | **MLP** | **TailDiff (Proposed, Ours)** | **0.3102** | **0.1720** | **0.1620** | **0.1899** | 0.1957 |
| **S&P 500** | **XGBoost** | No-Aug (Original) | 0.0000 | 0.0000 | 0.0000 | 0.1296 | **0.0173** |
| **S&P 500** | **XGBoost** | **TailDiff (Proposed, Ours)** | **0.5113** | **0.2310** | **0.2108** | **0.2626** | 0.3648 |
| **S&P 500** | **LightGBM** | **TailDiff (Proposed, Ours)** | **0.4958** | **0.2280** | **0.2096** | **0.2735** | 0.3388 |
| **S&P 500** | **MLP** | **TailDiff (Proposed, Ours)** | **0.4067** | **0.2150** | **0.2036** | **0.2437** | 0.3444 |

*Note: In all models, gains in Tail Recall and F1-Score are statistically significant ($p = 0.0039 < 0.01$ under Wilcoxon signed-rank test).*

---

## 📁 Repository Structure

```
TailDiff/
├── 复现包/ (src/)
│   ├── config.py                     # Central configuration & hyperparameters
│   ├── data_loader.py                # Causal market data loader & DaR 97.5% labeling
│   ├── models/
│   │   ├── diffusion.py              # 1D Causal TCN + FiLM + 20-step DDIM ODE Sampler
│   │   ├── dcr_gating.py             # DCR Leave-One-Out bilateral sweet-spot gating
│   │   ├── stylized_facts.py         # Financial stylized facts quality control suite
│   │   └── downstream.py             # 35-Fold Purged & Embargoed walk-forward evaluation
│   ├── baselines.py                  # SMOTE, Borderline-SMOTE, TimeGAN, C-VAE
│   ├── ablation.py                   # Gating variants & DDIM Pareto latency scan
│   ├── run_step1_main.py             # Step 1 master pipeline (CSI 300 & S&P 500)
│   ├── run_step2_ablation.py         # Step 2 ablation & Wilcoxon significance tests
│   └── build_all_publication_assets.py # Master exporter (Figures 1-6 & Tables 1-5)
│
├── 表格与图片/
│   ├── paper.tex                     # Full publication LaTeX paper (Elsevier format)
│   ├── references.bib                # BibTeX references database (28+ entries)
│   ├── figures/                      # 300 DPI Nature-palette publication figures (Fig 1-6)
│   └── tables/                       # Publication tables (Markdown & LaTeX format)
│
├── 原始数据/
│   ├── csi300_daily.csv              # CSI 300 daily index data (2015-2026)
│   └── sp500_daily.csv               # S&P 500 daily index data (2015-2026)
│
├── README.md
└── .gitignore
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/hu-zhixuan/TailDiff.git
cd TailDiff
pip install torch torchvision numpy pandas scikit-learn lightgbm xgboost matplotlib scipy
```

### 2. Run Step 1 Master Benchmark
```bash
python -m 复现包.run_step1_main
```

### 3. Run Step 2 Systematic Ablation & Wilcoxon Tests
```bash
python -m 复现包.run_step2_ablation
```

### 4. Export All Publication Figures & LaTeX Tables
```bash
python -m 复现包.build_all_publication_assets
```

---

## 📄 Paper Compilation (LaTeX)

The complete publication manuscript is written in the standard Elsevier format (`elsarticle.cls` with `\linenumbers`):
```bash
cd 表格与图片
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## 📑 Citation

If you find TailDiff useful for your research, please cite:

```bibtex
@article{hu2026taildiff,
  title={TailDiff: A Lightweight Conditional Diffusion Modeling Approach for Financial Tail-Risk Early Warning},
  author={Hu, Zhixuan and Collaborators},
  journal={Expert Systems with Applications},
  year={2026}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
