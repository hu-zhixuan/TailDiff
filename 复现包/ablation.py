"""
Ablation Suite for TailDiff:
1. Gating Mechanism Ablation: (No-Aug, Replicate, Full-Synth, Drop-Near, Drop-Far, Bilateral)
2. DDIM Step & Latency-Fidelity Trade-off Analysis (S in {5, 10, 20, 50, 100})
3. Quantile Threshold Sensitivity Analysis (Q1/Q99, Q5/Q95, Q10/Q90)
"""
import time
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from scipy.stats import wasserstein_distance

from src.config import TailDiffConfig
from src.models.diffusion import DiffusionSampler, EpsNet
from src.models.downstream import evaluate_walk_forward


def build_ablation_datasets(
    returns: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    X_synth: np.ndarray,
    source_idx: np.ndarray,
    X_crash: np.ndarray,
    d_crash: np.ndarray,
    loo: np.ndarray,
    d_synth: np.ndarray,
    n_survived: int,
    q_lo: float = 0.05,
    q_hi: float = 0.95
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    构造系统消融数据集集合：
    - real: 仅真实样本
    - replicate: 等量复制真实暴跌样本 (同扩增量的纯复制基线)
    - full: 全量合成样本 (无门控)
    - drop-near: 仅剔除记忆样本 (DCR >= Q_lo)
    - drop-far: 仅剔除幻觉噪声 (DCR <= Q_hi)
    - bilateral: 双边门控 (Q_lo <= DCR <= Q_hi，本文方法)
    """
    lo_thr = np.quantile(loo, q_lo)
    hi_thr = np.quantile(loo, q_hi)
    d_synth_dates = d_crash[source_idx]

    modes = {"No-Aug (Real Only)": (returns, labels, dates)}

    # 同量复制对照组
    if n_survived > 0:
        rng = np.random.default_rng(42)
        rep_idx = rng.choice(len(X_crash), size=n_survived, replace=True)
        modes["Replicate Baseline"] = (
            np.concatenate([returns, X_crash[rep_idx]], axis=0),
            np.concatenate([labels, np.ones(n_survived, dtype=int)], axis=0),
            np.concatenate([dates, d_crash[rep_idx]], axis=0)
        )

    # 门控变体
    gating_masks = {
        "Full-Synth (No Gating)": np.ones(len(X_synth), dtype=bool),
        "Drop-Near Only (Anti-Memorization)": (d_synth >= lo_thr),
        "Drop-Far Only (Anti-Hallucination)": (d_synth <= hi_thr),
        "TailDiff Bilateral (Ours)": (d_synth >= lo_thr) & (d_synth <= hi_thr),
    }

    for name, mask in gating_masks.items():
        if mask.sum() > 0:
            modes[name] = (
                np.concatenate([returns, X_synth[mask]], axis=0),
                np.concatenate([labels, np.ones(int(mask.sum()), dtype=int)], axis=0),
                np.concatenate([dates, d_synth_dates[mask]], axis=0)
            )

    return modes


def evaluate_ddim_tradeoff(
    eps_net: EpsNet,
    sampler: DiffusionSampler,
    X_crash: np.ndarray,
    rv_crash: np.ndarray,
    step_candidates: List[int] = [5, 10, 20, 50, 100]
) -> List[Dict[str, Any]]:
    """评估 DDIM 步数对采样延迟 (ms) 与 Wasserstein-1 生成保真度的影响。"""
    eps_net.eval()
    sig = X_crash.std(axis=1)
    results = []

    for s in step_candidates:
        # 预热
        _ = sampler.ddim_sample(eps_net, n=32, y=1, rv=rv_crash[:32], sigma_t=sig[:32], S=s)

        # 测速
        t0 = time.perf_counter()
        n_test = min(len(X_crash), 128)
        synth = sampler.ddim_sample(
            eps_net, n=n_test, y=1,
            rv=rv_crash[:n_test],
            sigma_t=sig[:n_test],
            S=s
        ).squeeze(1).cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0 / n_test

        w1 = float(wasserstein_distance(X_crash.ravel(), synth.ravel()))
        results.append({
            "ddim_steps": s,
            "latency_ms_per_sample": latency_ms,
            "wasserstein_1": w1
        })
        print(f"  [DDIM Step Ablation] S={s:3d} | Latency: {latency_ms:.2f} ms/sample | W1: {w1:.4f}")

    return results
