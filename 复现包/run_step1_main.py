"""
Step 1 Master Pipeline: End-to-End Verification on Authentic CSI 300 and S&P 500 Market Data.
Executes:
1. Data loading & causal labeling (DaR 97.5% quantile, p ≈ 4.5%)
2. Diffusion Generator training & 20-step deterministic DDIM sampling
3. DCR Bilateral Quality Gating ([Q5%, Q95%] sweet spot) & bootstrap CI
4. Stylized facts statistical validation (Kurtosis, KS, Wasserstein-1, Vol Decay)
5. Purged & Embargoed Expanding-Window Walk-Forward CV across XGBoost, LightGBM, and MLP
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import TailDiffConfig
from src.data_loader import MarketDataLoader
from src.models.diffusion import EpsNet, DiffusionSampler, train_diffusion
from src.models.dcr_gating import dcr_distances, gate_from_distances, bootstrap_dcr_ci
from src.models.stylized_facts import stylized_facts_report
from src.models.downstream import evaluate_walk_forward


def run_taildiff_on_market(market_name: str, csv_path: str, cfg: TailDiffConfig):
    print(f"\n{'='*70}")
    print(f"🚀 [STEP 1] Running TailDiff on Market: {market_name}")
    print(f"{'='*70}")

    device = torch.device(cfg.device)
    loader = MarketDataLoader(cfg)
    df_raw = loader.load_raw_csv(csv_path)
    returns, dates, labels, rv20, meta = loader.create_tail_dataset(df_raw, market_name)

    n_samples = len(returns)
    n_crash = int(np.sum(labels == 1))
    print(f"[*] Total Samples: {n_samples}, Tail-Risk Events: {n_crash} ({meta['crash_prevalence']*100:.2f}%)")

    # ── Phase 1: 训练条件扩散生成器 ──
    print(f"\n--- [Phase 1: Conditional Diffusion Training & DDIM Sampling] ---")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ds = TensorDataset(
        torch.tensor(returns, dtype=torch.float32),
        torch.tensor(rv20, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    data_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    eps_net = EpsNet(cfg).to(device)
    n_params = sum(p.numel() for p in eps_net.parameters())
    print(f"[*] Lightweight Generator Params: {n_params / 1e6:.3f}M ({n_params:,} parameters)")

    opt = torch.optim.AdamW(eps_net.parameters(), lr=cfg.lr, weight_decay=1e-5)
    sampler = DiffusionSampler(cfg, device)

    train_diffusion(eps_net, data_loader, opt, sampler, cfg, epochs=cfg.epochs)

    # 采样极端暴跌候选池 (超额 R 倍)
    crash_mask = (labels == 1)
    X_crash = returns[crash_mask]
    rv_crash = rv20[crash_mask]
    d_crash = dates[crash_mask]

    R = cfg.oversample_ratio
    print(f"[*] Oversampling crash pool with R={R}x (Candidate pool size: {R * len(X_crash)})...")
    eps_net.eval()
    sig_crash = X_crash.std(axis=1)

    X_synth = sampler.ddim_sample(
        eps_net,
        n=R * len(X_crash),
        y=1,
        rv=np.repeat(rv_crash, R),
        sigma_t=torch.tensor(np.repeat(sig_crash, R), dtype=torch.float32),
        S=cfg.ddim_steps
    ).squeeze(1).cpu().numpy()

    source_idx = np.repeat(np.arange(len(X_crash)), R)
    d_synth_dates = d_crash[source_idx]

    # ── Phase 2: DCR 双边门控过滤 ──
    print(f"\n--- [Phase 2: DCR Bilateral Quality Gating] ---")
    loo_real, d_synth = dcr_distances(X_synth, X_crash)
    mask_kept, lam_eff, lo_thr, hi_thr = gate_from_distances(d_synth, loo_real, cfg.dcr_lo_q, cfg.dcr_hi_q)

    boot_ci = bootstrap_dcr_ci(loo_real, cfg.dcr_lo_q, cfg.dcr_hi_q, n_boot=500, seed=cfg.seed)
    print(f"[*] LOO Real Distance Range: [{np.min(loo_real):.4f}, {np.max(loo_real):.4f}], Median: {np.median(loo_real):.4f}")
    print(f"[*] DCR Sweet Spot [Q5%, Q95%]: [{lo_thr:.4f}, {hi_thr:.4f}]")
    print(f"[*] Bootstrap 95% CI: Lower {boot_ci['lo_ci_95']}, Upper {boot_ci['hi_ci_95']}")
    print(f"[*] Gating Filter Results: {len(X_synth)} candidates -> {int(mask_kept.sum())} survived")
    print(f"[*] Effective Augmentation Ratio λ_eff = {lam_eff:.2f}x (Emergent, not manually fixed)")

    X_kept = X_synth[mask_kept]
    d_kept = d_synth_dates[mask_kept]

    # ── Phase 3: 典型事实统计检验 ──
    print(f"\n--- [Phase 3: Financial Stylized Facts & Fidelity Validation] ---")
    rep_crash = stylized_facts_report(X_crash, X_kept)
    print(f"  [Crash Class] Excess Kurtosis: Real={rep_crash['excess_kurtosis_real']:.2f}, Synth={rep_crash['excess_kurtosis_synth']:.2f} (Delta={rep_crash['kurtosis_delta']:.2f})")
    print(f"  [Crash Class] Skewness: Real={rep_crash['skewness_real']:.2f}, Synth={rep_crash['skewness_synth']:.2f} (Delta={rep_crash['skewness_delta']:.2f})")
    print(f"  [Crash Class] 1-Wasserstein Distance: {rep_crash['wasserstein_1']:.4f}")
    print(f"  [Crash Class] KS-Statistic: {rep_crash['ks_stat']:.4f} (p={rep_crash['ks_pvalue']:.4e})")
    print(f"  [Crash Class] Volatility Decay L1 Dev: {rep_crash['vol_decay_l1']:.4f}")

    # 非暴跌类控制组检验 (y=0)
    normal_idx = np.where(labels == 0)[0]
    sub_norm = np.random.default_rng(0).choice(normal_idx, size=min(len(normal_idx), 500), replace=False)
    X0_synth = sampler.ddim_sample(
        eps_net,
        n=len(sub_norm),
        y=0,
        rv=rv20[sub_norm],
        sigma_t=torch.tensor(returns[sub_norm].std(axis=1), dtype=torch.float32),
        S=cfg.ddim_steps
    ).squeeze(1).cpu().numpy()
    rep_norm = stylized_facts_report(returns[sub_norm], X0_synth)
    print(f"  [Normal Class Control] Excess Kurtosis: Real={rep_norm['excess_kurtosis_real']:.2f}, Synth={rep_norm['excess_kurtosis_synth']:.2f}")
    print(f"  [Normal Class Control] 1-Wasserstein Distance: {rep_norm['wasserstein_1']:.4f}")

    # ── Phase 4: Purged Expanding-Window Walk-Forward 评测 ──
    print(f"\n--- [Phase 4: Purged & Embargoed Walk-Forward Downstream Evaluation] ---")
    # 构造增强数据集
    X_aug = np.concatenate([returns, X_kept], axis=0)
    y_aug = np.concatenate([labels, np.ones(len(X_kept), dtype=int)], axis=0)
    d_aug = np.concatenate([dates, d_kept], axis=0)

    # 对照组：同量简单复制 (Replicate)
    rng = np.random.default_rng(0)
    rep_idx = rng.choice(len(X_crash), size=len(X_kept), replace=True)
    X_rep = np.concatenate([returns, X_crash[rep_idx]], axis=0)
    y_rep = np.concatenate([labels, np.ones(len(X_kept), dtype=int)], axis=0)
    d_rep = np.concatenate([dates, d_crash[rep_idx]], axis=0)

    results_table = {}
    models_to_test = ["xgb", "lgbm", "mlp"]

    for model_name in models_to_test:
        try:
            # 1. No-Aug Baseline
            res_noaug = evaluate_walk_forward(
                returns, labels, dates, n_real=n_samples,
                H=cfg.horizon, embargo=cfg.embargo_days,
                min_train=cfg.cv_min_train, step=cfg.cv_step,
                model_kind=model_name, seed=cfg.seed
            )
            # 2. Replicate Baseline
            res_rep = evaluate_walk_forward(
                X_rep, y_rep, d_rep, n_real=n_samples,
                H=cfg.horizon, embargo=cfg.embargo_days,
                min_train=cfg.cv_min_train, step=cfg.cv_step,
                model_kind=model_name, seed=cfg.seed
            )
            # 3. TailDiff (Ours)
            res_taildiff = evaluate_walk_forward(
                X_aug, y_aug, d_aug, n_real=n_samples,
                H=cfg.horizon, embargo=cfg.embargo_days,
                min_train=cfg.cv_min_train, step=cfg.cv_step,
                model_kind=model_name, seed=cfg.seed
            )

            results_table[f"{model_name}_NoAug"] = res_noaug
            results_table[f"{model_name}_Replicate"] = res_rep
            results_table[f"{model_name}_TailDiff"] = res_taildiff

            print(f"  [{model_name.upper()}] No-Aug   -> Recall: {res_noaug['recall_mean']:.4f}±{res_noaug['recall_std']:.4f}, "
                  f"F1: {res_noaug['f1_mean']:.4f}±{res_noaug['f1_std']:.4f}, PR-AUC: {res_noaug['pr_auc_mean']:.4f}±{res_noaug['pr_auc_std']:.4f}, FPR: {res_noaug['fpr_mean']:.4f}")
            print(f"  [{model_name.upper()}] Replicate-> Recall: {res_rep['recall_mean']:.4f}±{res_rep['recall_std']:.4f}, "
                  f"F1: {res_rep['f1_mean']:.4f}±{res_rep['f1_std']:.4f}, PR-AUC: {res_rep['pr_auc_mean']:.4f}±{res_rep['pr_auc_std']:.4f}, FPR: {res_rep['fpr_mean']:.4f}")
            print(f"  [{model_name.upper()}] TailDiff -> Recall: {res_taildiff['recall_mean']:.4f}±{res_taildiff['recall_std']:.4f}, "
                  f"F1: {res_taildiff['f1_mean']:.4f}±{res_taildiff['f1_std']:.4f}, PR-AUC: {res_taildiff['pr_auc_mean']:.4f}±{res_taildiff['pr_auc_std']:.4f}, FPR: {res_taildiff['fpr_mean']:.4f}")
            
            f1_gain = ((res_taildiff['f1_mean'] - res_noaug['f1_mean']) / max(res_noaug['f1_mean'], 1e-4)) * 100
            prauc_gain = ((res_taildiff['pr_auc_mean'] - res_noaug['pr_auc_mean']) / max(res_noaug['pr_auc_mean'], 1e-4)) * 100
            print(f"  --> [Gain on {model_name.upper()}] F1 Gain: +{f1_gain:.1f}%, PR-AUC Gain: +{prauc_gain:.1f}%\n")

        except Exception as e:
            print(f"  [{model_name.upper()}] Error evaluating: {e}")

    summary = {
        "market": market_name,
        "meta": meta,
        "n_params": n_params,
        "dcr": {
            "lo_thr": lo_thr,
            "hi_thr": hi_thr,
            "lam_eff": lam_eff,
            "boot_ci": boot_ci
        },
        "stylized_facts": {
            "kurt_real": rep_crash["excess_kurtosis_real"],
            "kurt_synth": rep_crash["excess_kurtosis_synth"],
            "w1": rep_crash["wasserstein_1"],
            "ks_stat": rep_crash["ks_stat"],
            "vol_decay_l1": rep_crash["vol_decay_l1"]
        },
        "downstream": results_table
    }
    return summary


def main():
    cfg = TailDiffConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)
    os.makedirs(cfg.tables_dir, exist_ok=True)

    # 1. 跑 CSI 300
    summary_csi = run_taildiff_on_market("CSI300", cfg.csi300_path, cfg)
    
    # 2. 跑 S&P 500
    summary_sp = run_taildiff_on_market("SP500", cfg.sp500_path, cfg)

    # 保存 Step 1 结果
    output_path = os.path.join(cfg.output_dir, "step1_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert numpy types to native python for JSON serialization
        def default_serializer(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            raise TypeError(f"Unserializable object: {obj}")
        
        json.dump({"csi300": summary_csi, "sp500": summary_sp}, f, indent=2, default=default_serializer)
    
    print(f"\n🎉 [STEP 1 COMPLETED] All results saved to {output_path}!")


if __name__ == "__main__":
    main()
