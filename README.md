# TailDiff: A Lightweight Conditional Diffusion Modeling Approach for Financial Tail-Risk Early Warning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code, data and result tables for the TailDiff experiments. The accompanying manuscript is not included in this repository.

---

## Overview

Market crashes are rare, so classifiers trained on daily index data almost never predict them. TailDiff tries to fix this by generating additional synthetic "pre-crash" windows with a small conditional diffusion model, filtering them, and adding them to the training set of ordinary downstream classifiers (XGBoost / LightGBM / MLP).

Main components:

- **Generator:** a 1D dilated causal convolution network (about 0.91M parameters, receptive field 37 ≥ window length 32), conditioned on 20-day realized volatility via FiLM. Sampling uses 20-step deterministic DDIM (reported at about 5 ms per sample on CPU).
- **Quality gate (DCR):** keeps a synthetic sample only if its nearest-neighbour distance to the real crash windows falls between the 5th and 95th percentiles of the real samples' own leave-one-out distances. Too close suggests memorisation; too far suggests an implausible sample.
- **Evaluation:** expanding-window walk-forward CV with purging and a 5-day embargo (about 35 folds per market). Synthetic samples are assigned the date of the real window they were derived from and obey the same purge rule.
- **Labels:** a day is labelled as a tail event if the forward drawdown exceeds the EVT-estimated 97.5% DaR.

![TailDiff Architecture](./表格与图片/figures/Fig1_TailDiff_Architecture.png)

---

## Results (walk-forward CV, mean over folds)

| Market | Classifier | Training data | Recall | Precision | F1 | PR-AUC | FPR |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| CSI 300 | XGBoost | Real only | 0.009 | 0.051 | 0.015 | 0.140 | 0.012 |
| CSI 300 | XGBoost | Real + replicated positives | 0.014 | 0.058 | 0.024 | 0.141 | 0.010 |
| CSI 300 | XGBoost | Real + TailDiff | 0.202 | 0.186 | 0.136 | 0.177 | 0.194 |
| CSI 300 | LightGBM | Real + TailDiff | 0.210 | 0.194 | 0.138 | 0.177 | 0.199 |
| CSI 300 | MLP | Real + TailDiff | 0.310 | 0.172 | 0.162 | 0.190 | 0.196 |
| S&P 500 | XGBoost | Real only | 0.000 | 0.000 | 0.000 | 0.130 | 0.017 |
| S&P 500 | XGBoost | Real + TailDiff | 0.511 | 0.231 | 0.211 | 0.263 | 0.365 |
| S&P 500 | LightGBM | Real + TailDiff | 0.496 | 0.228 | 0.210 | 0.274 | 0.339 |
| S&P 500 | MLP | Real + TailDiff | 0.407 | 0.215 | 0.204 | 0.244 | 0.344 |

Full tables with standard deviations are in `表格与图片/tables/`.

**How to read these numbers:**

- Without augmentation the classifiers essentially never predict a crash (recall ≈ 0), so large *relative* recall gains mostly reflect moving away from that degenerate state.
- Higher recall comes with a much higher false positive rate (e.g. 0.017 → 0.365 on S&P 500 with XGBoost). Much of the recall gain is a shift in the operating point rather than better ranking.
- PR-AUC, which does not depend on the threshold, improves modestly. In the one-sided Wilcoxon test against real-only training, the PR-AUC difference is **not significant** (p = 0.72); against replicated positives it is borderline (p = 0.048).
- The recall/F1 test (W = 36, p = 0.0039) is the smallest p-value possible with 8 non-tied pairs: most folds had identical results under both methods (often no tail event or zero recall for both) and were dropped by the test. It should not be read as "significant across 35 folds".
- Standard deviations across folds are large (recall ± 0.3–0.4).

---

## Repository Structure

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
│   ├── baselines.py                  # SMOTE, Borderline-SMOTE, C-VAE
│   ├── ablation.py                   # Gating variants & DDIM Pareto latency scan
│   ├── run_step1_main.py             # Step 1 master pipeline (CSI 300 & S&P 500)
│   ├── run_step2_ablation.py         # Step 2 ablation & Wilcoxon significance tests
│   └── build_all_publication_assets.py # Exports Figures 1-6 & Tables 1-5
│
├── 表格与图片/
│   ├── figures/                      # Figures 1-6
│   └── tables/                       # Tables 1-5 (Markdown)
│
├── 原始数据/
│   ├── csi300_daily.csv              # CSI 300 daily index data (2015-2026)
│   └── sp500_daily.csv               # S&P 500 daily index data (2015-2026)
│
├── README.md
└── .gitignore
```

---

## Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/hu-zhixuan/TailDiff.git
cd TailDiff
pip install torch torchvision numpy pandas scikit-learn lightgbm xgboost matplotlib scipy
```

### 2. Main experiment
```bash
python -m 复现包.run_step1_main
```

### 3. Ablation and Wilcoxon tests
```bash
python -m 复现包.run_step2_ablation
```

### 4. Export figures and tables
```bash
python -m 复现包.build_all_publication_assets
```

---

## Limitations

- Only two markets (CSI 300 and S&P 500 daily index data, 2015–2026) with a few dozen tail episodes each; results may not transfer to other assets or frequencies.
- The downstream classifiers use a fixed decision threshold. A fairer comparison would tune the real-only baseline's threshold for the same FPR, which has not been done yet.
- The C-VAE and SMOTE baselines are implemented in `baselines.py` but are not in the main table above. TimeGAN is not implemented.
- The generator does not reproduce fat tails well: the kurtosis recorded in `save_summary.py` is about 2.7 for synthetic windows vs about 10.5 for real crash windows.

### Note on the figures and tables in `表格与图片/`

`复现包/build_all_publication_assets.py` does **not** compute its outputs from experiment runs:

- **Tables 1–5 and Figures 3–5** are written from numbers hard-coded in the script. To check them, re-run `run_step1_main.py` / `run_step2_ablation.py` and compare.
- **Figure 2** uses placeholder data: "synthetic" samples are real crash windows plus Gaussian noise, and the DCR distances are random draws, not outputs of the diffusion model.
- **Figure 6** (crash warning timeline and hedging backtest) is **fully simulated**: a synthetic price path with injected crashes and warning probabilities placed on those crashes. The drawdown numbers in its legend (−46.5% vs −14.2%) are not results.

These figures should be regenerated from real model outputs, or removed, before being used anywhere.

---

## License

MIT
