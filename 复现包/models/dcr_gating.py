"""
Phase 2: DCR (Distance to Closest Record) Bilateral Quality Gating Engine.
Ensures Non-Memorization (< Q5%) and Anti-Hallucination (> Q95%) for Synthetic Tail Events.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, Any


def loo_dcr_baseline(X_real: np.ndarray) -> np.ndarray:
    """
    留一法 (Leave-One-Out) 计算真实极端样本之间的互测 DCR 经验分布。
    d_LOO[i] = min_{j != i} ||x_real[i] - x_real[j]||_2
    """
    n = len(X_real)
    dcr = np.zeros(n, dtype=np.float32)
    for i in range(n):
        others = np.delete(X_real, i, axis=0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(others)
        dcr[i] = nbrs.kneighbors(X_real[i : i + 1])[0][0, 0]
    return dcr


def dcr_distances(X_synth: np.ndarray, X_real: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算:
    1. 真实暴跌集的 LOO 内部距离分布
    2. 每条合成暴跌样本到真实暴跌集的最近邻 L2 距离 d_synth
    """
    loo = loo_dcr_baseline(X_real)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_real)
    d_synth = nbrs.kneighbors(X_synth)[0][:, 0]
    return loo, d_synth


def gate_from_distances(d_synth: np.ndarray, loo: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> Tuple[np.ndarray, float, float, float]:
    """
    双边甜蜜区门控：
    - d_synth < Q_lo(LOO): 过于接近真实数据，判定为机械背诵/过拟合 -> 丢弃
    - d_synth > Q_hi(LOO): 偏离真实流形太远，判定为虚构幻觉/离群噪声 -> 丢弃
    - 仅保留 [Q_lo, Q_hi] 区间内的优质多样性样本
    """
    lo_thr = float(np.quantile(loo, q_lo))
    hi_thr = float(np.quantile(loo, q_hi))
    mask = (d_synth >= lo_thr) & (d_synth <= hi_thr)
    lam_eff = float(mask.sum()) / max(len(loo), 1)
    return mask, lam_eff, lo_thr, hi_thr


def bootstrap_dcr_ci(loo: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95, n_boot: int = 500, seed: int = 0) -> Dict[str, Tuple[float, float]]:
    """针对小样本暴跌事件，使用 Bootstrap 计算门控阈值的 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    n = len(loo)
    lo_boots = []
    hi_boots = []
    for _ in range(n_boot):
        sample = rng.choice(loo, size=n, replace=True)
        lo_boots.append(np.quantile(sample, q_lo))
        hi_boots.append(np.quantile(sample, q_hi))

    return {
        "lo_ci_95": (float(np.percentile(lo_boots, 2.5)), float(np.percentile(lo_boots, 97.5))),
        "hi_ci_95": (float(np.percentile(hi_boots, 2.5)), float(np.percentile(hi_boots, 97.5))),
        "lo_mean": float(np.mean(lo_boots)),
        "hi_mean": float(np.mean(hi_boots)),
    }
