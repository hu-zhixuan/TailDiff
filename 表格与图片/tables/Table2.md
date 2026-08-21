# Table 2. Quantitative Evaluation of Synthetic Tail-Risk Generation Quality (CSI 300 & S&P 500)

| Market Target | Generation Method | KS Stat (↓) | Wasserstein-1 (↓) | Kurtosis Delta (↓) | Vol Decay L1 (↓) | Non-Memorization |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CSI 300 Benchmark** | **Real Benchmark** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Verified |
| **CSI 300** | **SMOTE (Linear Interpolation)** | 0.2415 | 0.0982 | 3.8415 | 0.1840 | Failed (< Q5%) |
| **CSI 300** | **TimeGAN (Adversarial GAN)** | 0.1842 | 0.0654 | 2.1054 | 0.1210 | Marginal |
| **CSI 300** | **TailDiff (DDIM-20, Ours)** | **0.0794** | **0.0036** | **0.2300** | **0.0673** | **Verified (Passed)** |
| **S&P 500 Benchmark** | **Real Benchmark** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Verified |
| **S&P 500** | **TailDiff (DDIM-20, Ours)** | **0.0680** | **0.0018** | **7.8100** | **0.0635** | **Verified (Passed)** |
