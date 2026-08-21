"""
Phase 3: Quantitative Statistical Fidelity & Financial Stylized Facts Validation Suite.
Validates:
1. Fat-Tails (Excess Kurtosis & Skewness Deltas)
2. Non-Gaussianity & Goodness-of-Fit (Kolmogorov-Smirnov Test)
3. Volatility Clustering Decay Dynamics (ACF of Absolute Returns)
4. Manifold Earth Mover's Distance (1-Wasserstein Distance)
"""
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from typing import Dict, Any


def vol_clustering_decay(r_flat: np.ndarray, max_lag: int = 15) -> np.ndarray:
    """计算绝对收益率序列的自相关函数 ACF(|r_t|, |r_{t+k}|)。"""
    abs_r = np.abs(r_flat - np.mean(r_flat))
    decay = [1.0]
    n = len(abs_r)
    for k in range(1, max_lag + 1):
        if k >= n:
            decay.append(0.0)
        else:
            corr = np.corrcoef(abs_r[:-k], abs_r[k:])[0, 1]
            decay.append(0.0 if np.isnan(corr) else float(corr))
    return np.array(decay, dtype=np.float32)


def stylized_facts_report(real_data: np.ndarray, synth_data: np.ndarray, max_lag: int = 15) -> Dict[str, Any]:
    """
    全方位金融典型事实保真度量化评测。
    real_data, synth_data 可以是 (N, L) 二维矩阵或 (N*L,) 展平数组。
    """
    r_real = real_data.ravel()
    r_synth = synth_data.ravel()

    # 1. 尖峰厚尾与偏度
    kurt_real = float(stats.kurtosis(r_real, fisher=True))
    kurt_synth = float(stats.kurtosis(r_synth, fisher=True))
    skew_real = float(stats.skew(r_real))
    skew_synth = float(stats.skew(r_synth))

    kurt_delta = abs(kurt_synth - kurt_real)
    skew_delta = abs(skew_synth - skew_real)

    # 2. 1-Wasserstein 距离 (分布重叠度)
    w1_dist = float(wasserstein_distance(r_real, r_synth))

    # 3. 标准化后的 KS 检验
    z_synth = (r_synth - np.mean(r_synth)) / (np.std(r_synth) + 1e-10)
    ks_res = stats.kstest(z_synth, "norm")
    ks_stat = float(ks_res.statistic)
    ks_pvalue = float(ks_res.pvalue)

    # 4. 波动率聚集衰减曲线 L1 偏差
    decay_real = vol_clustering_decay(r_real, max_lag=max_lag)
    decay_synth = vol_clustering_decay(r_synth, max_lag=max_lag)
    vol_decay_l1 = float(np.mean(np.abs(decay_real - decay_synth)))

    return {
        "excess_kurtosis_real": kurt_real,
        "excess_kurtosis_synth": kurt_synth,
        "kurtosis_delta": kurt_delta,
        "skewness_real": skew_real,
        "skewness_synth": skew_synth,
        "skewness_delta": skew_delta,
        "wasserstein_1": w1_dist,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "vol_decay_l1": vol_decay_l1,
        "decay_curve_real": decay_real,
        "decay_curve_synth": decay_synth,
    }
