"""
Global Configuration for TailDiff Empirical Experiments and Benchmark Suite.
"""
from dataclasses import dataclass, field
import os
import torch

@dataclass
class TailDiffConfig:
    # ── 数据与序列配置 ──
    L: int = 32                     # 历史对数收益率窗口长度 (Lookback window)
    horizon: int = 10               # 前瞻风险评估窗口天数 (Forward risk horizon H)
    rv_window: int = 20             # 已实现波动率 RV 滚动窗口天数
    crash_quantile: float = 0.045   # 极端风险发生率目标 (Top 4.5% tail event, p ≈ 4.5%)
    crash_threshold: float = -0.08  # 极端暴跌硬阈值 (例如 10日累计收益 <= -8%)
    use_quantile_label: bool = True # True 使用历史分位数打标，False 使用硬阈值

    # ── 扩散模型超参 ──
    num_diffusion_steps: int = 200  # 训练用扩散总步数 T
    ddim_steps: int = 20            # 采样用确定性 DDIM 步数 S (Fast ODE)
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # ── 1D 膨胀因果 TCN 骨干 ──
    # 6 个残差块，RF = 1 + (3-1) * 2 * (1+2+4+8+1+2) = 73 >= L=32
    channels: int = 128
    dilations: tuple = (1, 2, 4, 8, 1, 2)
    kernel_size: int = 3
    num_groups: int = 16            # GroupNorm 分组数
    cond_dim: int = 64 + 1 + 32     # time_emb(64) + log_rv(1) + label_emb(32)

    # ── 训练与优化 ──
    lr: float = 2e-4
    batch_size: int = 256
    epochs: int = 250
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ── DCR 双边门控 ──
    oversample_ratio: int = 20      # 超额生成倍率候选池 R
    dcr_lo_q: float = 0.05          # 留一基线下界 (防机械背诵/记忆)
    dcr_hi_q: float = 0.95          # 留一基线上界 (防虚构幻觉/离群噪声)

    # ── Purged & Embargoed Expanding-Window CV ──
    embargo_days: int = 5           # 隔离期 (Embargo gap)
    cv_min_train: int = 500         # 初始最小训练样本数 (~2年)
    cv_step: int = 60               # 前向滚动步长 (约每季度/60个交易日)

    # ── 路径配置 ──
    csi300_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "原始数据", "csi300_daily.csv")
    sp500_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "原始数据", "sp500_daily.csv")
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    figures_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "figures")
    tables_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "tables")
