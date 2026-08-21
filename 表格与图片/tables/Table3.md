# Table 3. Downstream Risk Warning Performance across Baselines and Models (35-Fold Purged CV, p ≈ 4.5%)

| Classifier | Augmentation / Baseline Strategy | Tail Recall (↑) | Precision (↑) | F1-Score (↑) | PR-AUC (↑) | False Alarm FPR (↓) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (CSI 300)** | **No-Aug (Original Imbalanced)** | 0.0093 ± 0.0307 | 0.0512 ± 0.0420 | 0.0152 ± 0.0503 | 0.1396 ± 0.1320 | **0.0116** |
| **XGBoost (CSI 300)** | **Replicate (Oversample Real)** | 0.0139 ± 0.0461 | 0.0580 ± 0.0410 | 0.0238 ± 0.0790 | 0.1412 ± 0.1366 | **0.0097** |
| **XGBoost (CSI 300)** | **TailDiff (Proposed, Ours)** | **0.2024 ± 0.3152** | **0.1860 ± 0.0620** | **0.1361 ± 0.2123** | **0.1773 ± 0.1911** | 0.1941 |
| **LightGBM (CSI 300)**| **No-Aug (Original Imbalanced)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1488 ± 0.1509 | **0.0016** |
| **LightGBM (CSI 300)**| **Replicate (Oversample Real)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1422 ± 0.1283 | **0.0000** |
| **LightGBM (CSI 300)**| **TailDiff (Proposed, Ours)** | **0.2097 ± 0.3116** | **0.1940 ± 0.0580** | **0.1379 ± 0.2082** | **0.1768 ± 0.1938** | 0.1986 |
| **MLP (CSI 300)**     | **No-Aug (Original Imbalanced)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1880 ± 0.1365 | **0.0000** |
| **MLP (CSI 300)**     | **Replicate (Oversample Real)** | 0.0139 ± 0.0461 | 0.0450 ± 0.0380 | 0.0200 ± 0.0663 | 0.1211 ± 0.0847 | 0.0623 |
| **MLP (CSI 300)**     | **TailDiff (Proposed, Ours)** | **0.3102 ± 0.3656** | **0.1720 ± 0.0640** | **0.1620 ± 0.1909** | **0.1899 ± 0.1691** | 0.1957 |
| **XGBoost (S&P 500)** | **No-Aug (Original Imbalanced)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1296 ± 0.0983 | **0.0173** |
| **XGBoost (S&P 500)** | **TailDiff (Proposed, Ours)** | **0.5113 ± 0.4119** | **0.2310 ± 0.0710** | **0.2108 ± 0.1862** | **0.2626 ± 0.2247** | 0.3648 |
| **LightGBM (S&P 500)**| **TailDiff (Proposed, Ours)** | **0.4958 ± 0.4234** | **0.2280 ± 0.0690** | **0.2096 ± 0.1854** | **0.2735 ± 0.2357** | 0.3388 |
| **MLP (S&P 500)**     | **TailDiff (Proposed, Ours)** | **0.4067 ± 0.3386** | **0.2150 ± 0.0650** | **0.2036 ± 0.1708** | **0.2437 ± 0.1851** | 0.3444 |

*Note: Wilcoxon signed-rank test across all folds: p = 0.0039 < 0.01 (***) vs. No-Aug on Recall and F1-Score.*
