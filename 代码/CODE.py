"""
CrashDiff: 条件扩散 + DCR 双边门控的暴跌预警数据扩增（单文件实现）
Phase 1 条件扩散生成 (TCN+FiLM+DDIM) / Phase 2 DCR 双边门控 / Phase 3 程式化事实检验 / Phase 4 Purged CV + 消融
依赖: torch numpy scipy scikit-learn xgboost pandas；lightgbm 可选（未安装自动跳过）

实验层修订（模型结构零改动，主流程默认行为不变）：
  · 消融套件 build_modes(): replicate 同量复制基线 / full 无门控 / 单边门控×2（复用同一批合成候选）
  · λ_eff 分位数敏感性扫描 (1/99, 5/95, 10/90)
  · DCR 阈值 bootstrap 95% CI（暴跌样本少时的不稳定度可视化）
  · y=0 生成质检（证明生成器学到整体动力学，不进下游）
  · perfold_robustness(): 可选 per-fold 生成器重训稳健性检验（默认关闭）
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import recall_score, average_precision_score, f1_score
from scipy import stats
from scipy.stats import wasserstein_distance
from xgboost import XGBClassifier
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
# 0. 超参
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # 数据
    L: int = 32                    # 历史窗口长度
    horizon: int = 5               # 未来 H 个交易日内是否暴跌（打标）
    # 扩散
    num_diffusion_steps: int = 200 # 训练用 T
    ddim_steps: int = 20           # 采样用确定性 DDIM 步数
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # TCN 骨干：6 个残差块 × 每块 2 层同膨胀因果卷积
    # RF = 1 + (k-1) × 2 × Σd = 1 + 2×2×18 = 73 ≥ L=32
    channels: int = 128
    dilations: tuple = (1, 2, 4, 8, 1, 2)
    kernel_size: int = 3
    num_groups: int = 16           # GroupNorm
    # 训练
    lr: float = 2e-4
    batch_size: int = 256
    epochs: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # DCR
    oversample_ratio: int = 20     # 超额生成倍率（候选池大小，非最终扩增倍率）
    dcr_lo_q: float = 0.05
    dcr_hi_q: float = 0.95
    # Purged CV（交易日语义：训练样本 i 需满足 i + H + embargo < 验证段起点）
    embargo_days: int = 5
    cv_min_train: int = 500
    cv_step: int = 60


# ═══════════════════════════════════════════════════════════════════
# 1. Phase 1: 条件扩散生成器（TCN + FiLM + DDIM）
# ═══════════════════════════════════════════════════════════════════
class CausalConv1d(nn.Module):
    """严格因果：只左填充，输出只依赖过去。"""
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):                     # (B, C, L)
        return self.conv(F.pad(x, (self.pad, 0)))


class FiLM(nn.Module):
    """h' = (1+γ)⊙h + β，条件 c 逐通道仿射调制 GroupNorm 后的特征。"""
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, h, c):                  # h:(B,C,L)  c:(B,cond_dim)
        g, b = self.to_gamma(c), self.to_beta(c)
        return (1 + g.unsqueeze(-1)) * h + b.unsqueeze(-1)


class ResDilatedBlock(nn.Module):
    """膨胀残差块：CausalConv → GN → FiLM → GELU ×2 + 残差。"""
    def __init__(self, C, k, d, gn_groups, cond_dim):
        super().__init__()
        self.conv1, self.conv2 = CausalConv1d(C, C, k, d), CausalConv1d(C, C, k, d)
        self.norm1, self.norm2 = nn.GroupNorm(gn_groups, C), nn.GroupNorm(gn_groups, C)
        self.film1, self.film2 = FiLM(cond_dim, C), FiLM(cond_dim, C)

    def forward(self, x, c):
        h = F.gelu(self.film1(self.norm1(self.conv1(x)), c))
        h = self.film2(self.norm2(self.conv2(h)), c)
        return F.gelu(x + h)


class EpsNet(nn.Module):
    """ε-预测网络。输入 x̃:(B,1,L)，条件 = sin-cos(t) ⊕ log(RV20) ⊕ label_emb。"""
    def __init__(self, cfg: Config):
        super().__init__()
        C = cfg.channels
        self.t_emb = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
        self.y_emb = nn.Embedding(2, 32)
        cond_dim = 64 + 1 + 32
        self.in_proj = CausalConv1d(1, C, 1, 1)
        self.blocks = nn.ModuleList(
            [ResDilatedBlock(C, cfg.kernel_size, d, cfg.num_groups, cond_dim)
             for d in cfg.dilations])
        self.out_proj = nn.Conv1d(C, 1, 1)

    @staticmethod
    def _t_sincos(t, dim=64):
        half = dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x, t, rv, y):
        B = x.size(0)
        c = torch.cat([
            self.t_emb(self._t_sincos(t)),               # (B,64)
            torch.log(rv.clamp(min=1e-8)).view(B, 1),    # (B,1)
            self.y_emb(y.long()),                        # (B,32)
        ], dim=-1)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out_proj(h)                          # ε:(B,1,L)


class DiffusionSampler:
    """训练：DDPM ε-目标；采样：DDIM 确定性 η=0。"""
    def __init__(self, cfg: Config, device):
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_diffusion_steps)
        self.alpha_bar = torch.cumprod(1 - betas, dim=0).to(device)
        self.cfg, self.device = cfg, device

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @staticmethod
    def _to_tensor(v, device):
        if not torch.is_tensor(v):
            v = torch.tensor(np.asarray(v, dtype=np.float32))
        return v.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def ddim_sample(self, eps_net, n, y, rv, sigma_t):
        """
        η=0 确定性 ODE，20 步。
        n: 样本数; y: 标量; rv: 标量或长度 n（每样本条件）;
        sigma_t: 长度 n（每样本局部波动率，反归一化用）。
        """
        S = self.cfg.ddim_steps
        taus = torch.linspace(self.cfg.num_diffusion_steps - 1, 0, S,
                              device=self.device).long()   # 递减时间步
        x = torch.randn(n, 1, self.cfg.L, device=self.device)
        y = torch.full((n,), int(y), device=self.device, dtype=torch.long)
        rv = self._to_tensor(rv, self.device)
        if rv.dim() == 0:
            rv = rv.expand(n)
        sigma_t = self._to_tensor(sigma_t, self.device)

        for i in range(S):
            t = taus[i]
            ab_t = self.alpha_bar[t].view(-1, 1, 1)
            eps = eps_net(x, t.expand(n), rv, y)
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if i < S - 1:                                   # 最后一步直接输出 x0
                ab_next = self.alpha_bar[taus[i + 1]].view(-1, 1, 1)
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
            else:
                x = x0
        return x * sigma_t.view(-1, 1, 1)                   # 反归一化 x̃ × σ


def train_diffusion(eps_net, loader, opt, sampler, cfg, epochs=None):
    """局部波动率归一化训练：x̃ = r / σ_t，去量纲学习标准形态。"""
    epochs = epochs or cfg.epochs
    for epoch in range(epochs):
        eps_net.train()
        tot = 0.0
        for (r, rv, y) in loader:
            r = r.to(cfg.device).unsqueeze(1)               # (B,L)→(B,1,L)
            rv, y = rv.to(cfg.device), y.to(cfg.device)
            sigma = r.std(dim=-1, keepdim=True).clamp(min=1e-4)   # (B,1,1)
            x_tilde = r / sigma
            t = torch.randint(0, cfg.num_diffusion_steps, (r.size(0),), device=cfg.device)
            noise = torch.randn_like(x_tilde)
            loss = F.mse_loss(eps_net(sampler.q_sample(x_tilde, t, noise), t, rv, y), noise)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  [diffusion] epoch {epoch+1}/{epochs}  loss={tot/len(loader):.4f}")


# ═══════════════════════════════════════════════════════════════════
# 2. Phase 2: DCR 双边门控过滤
# ═══════════════════════════════════════════════════════════════════
def loo_dcr_baseline(X_real):
    """留一互算真实样本间 DCR 分布，作为"甜蜜区"经验上下界。"""
    dcr = np.zeros(len(X_real))
    for i in range(len(X_real)):
        others = np.delete(X_real, i, axis=0)
        nbrs = NearestNeighbors(n_neighbors=1).fit(others)
        dcr[i] = nbrs.kneighbors(X_real[i:i + 1])[0][0, 0]
    return dcr


def dcr_distances(X_synth, X_real):
    """返回 (LOO 基线分布, 每条合成样本到真实集的最小 L2 距离)。"""
    loo = loo_dcr_baseline(X_real)
    d_synth = NearestNeighbors(n_neighbors=1).fit(X_real).kneighbors(X_synth)[0][:, 0]
    return loo, d_synth


def gate_from_distances(d_synth, loo, q_lo, q_hi):
    """双边门控：返回 (mask, λ_eff)。调用方自行切片。"""
    lo, hi = np.quantile(loo, q_lo), np.quantile(loo, q_hi)
    mask = (d_synth >= lo) & (d_synth <= hi)     # 太近=记忆，太远=幻觉
    return mask, mask.sum() / max(len(loo), 1)


# ═══════════════════════════════════════════════════════════════════
# 3. Phase 3: 程式化事实统计检验
# ═══════════════════════════════════════════════════════════════════
def vol_clustering_decay(r, max_lag=10):
    """Corr(|r_t|, |r_{t+k}|) 衰减曲线。"""
    abs_r = np.abs(r - r.mean())
    return np.array([1.0 if k == 0 else np.corrcoef(abs_r[:-k], abs_r[k:])[0, 1]
                     for k in range(max_lag + 1)])

def stylized_facts_report(real_flat, synth_flat, max_lag=10):
    """尖峰厚尾（超额峰度 + KS）+ 波动率聚集衰减 + Wasserstein-1。"""
    z_s = (synth_flat - synth_flat.mean()) / (synth_flat.std() + 1e-12)
    return {
        "excess_kurtosis_real": stats.kurtosis(real_flat, fisher=True),
        "excess_kurtosis_synth": stats.kurtosis(synth_flat, fisher=True),
        "ks_stat": stats.kstest(z_s, "norm")[0],
        "wasserstein_1": wasserstein_distance(real_flat, synth_flat),
        "vol_decay_l1": np.abs(vol_clustering_decay(real_flat, max_lag)
                               - vol_clustering_decay(synth_flat, max_lag)).mean(),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. Phase 4: 下游监督分类 + Purged Expanding-Window Walk-Forward CV
# ═══════════════════════════════════════════════════════════════════
def purged_expanding_walk_forward(n, H=5, embargo=5, min_train=500, step=60):
    """
    交易日语义的位置 purge：
    训练样本 i 的标签窗口 [i, i+H] 必须与验证段 [start, ...) 隔开至少 embargo 个交易日，
    即 i + H + embargo < start。yield (train_idx, valid_idx, start)。
    """
    start = min_train
    while start < n - 1:
        val_end = min(start + step, n)
        hi = start - H - embargo
        if hi > 0:
            yield np.arange(0, hi), np.arange(start, val_end), start
        start = val_end


def make_clf(kind, spw):
    """下游模型工厂：xgb / lgbm / mlp。spw = 负正样本比（类权重）。"""
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=spw, eval_metric="aucpr",
            n_jobs=-1, random_state=0)
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("lightgbm 未安装（pip install lightgbm），跳过该模型")
        return LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            scale_pos_weight=spw, n_jobs=-1, random_state=0, verbose=-1)
    if kind == "mlp":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        # MLP 不支持类权重：正类平衡由合成扩增天然提供
        return Pipeline([("sc", StandardScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64),
                                               max_iter=300, early_stopping=True,
                                               random_state=0))])
    raise ValueError(f"未知模型: {kind}")


def train_and_evaluate(X_aug, y_aug, d_aug, n_real, H=5, embargo=5,
                       min_train=500, step=60, model_kind="xgb"):
    """
    n_real: 前 n_real 条为真实样本，之后为合成样本。
    合成样本只进训练集，且其来源真实样本的位置 + H + embargo < 验证段起点——
    即合成样本与真实样本执行完全相同的 purge 规则，杜绝来源泄漏。
    验证集全部为真实样本。
    """
    dates_real = d_aug[:n_real]
    recalls, pr_aucs, f1s = [], [], []
    for tr_real, va, start in purged_expanding_walk_forward(
            n_real, H, embargo, min_train, step):
        if len(d_aug) > n_real:
            # 合成样本日期 → 在真实序列中的位置（来源日期 ⊂ 真实日期，精确映射）
            pos = np.searchsorted(dates_real, d_aug[n_real:], side="right") - 1
            synth_ok = np.where(pos + H + embargo < start)[0]
            tr = np.concatenate([tr_real, n_real + synth_ok])
        else:
            tr = tr_real
        X_tr, y_tr = X_aug[tr], y_aug[tr]
        X_va, y_va = X_aug[va], y_aug[va]
        if (y_tr == 1).sum() == 0 or (y_va == 1).sum() == 0:
            continue
        clf = make_clf(model_kind, (y_tr == 0).sum() / (y_tr == 1).sum())
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_va)[:, 1]
        yhat = (p > 0.5).astype(int)
        recalls.append(recall_score(y_va, yhat, zero_division=0))
        pr_aucs.append(average_precision_score(y_va, p))
        f1s.append(f1_score(y_va, yhat, zero_division=0))
    if not recalls:
        print(f"  [{model_kind}] 警告：没有可用 fold（验证段或训练段无正样本）")
        return None
    return {"Recall": np.mean(recalls), "PR-AUC": np.mean(pr_aucs),
            "F1": np.mean(f1s), "n_folds": len(recalls)}


# ═══════════════════════════════════════════════════════════════════
# 5. 数据管道
# ═══════════════════════════════════════════════════════════════════
def make_dataset(prices, L=32, H=5, rv_window=20, crash_threshold=-0.10):
    """
    prices: pd.Series，DatetimeIndex，日收盘价。
    防穿越：特征与 RV 只用截至 t 的信息，标签用 (t, t+H]（交易日语义）。
    末端 H 天未来未知的样本一律删除。
    """
    log_ret = np.log(prices).diff().dropna()
    rv20 = log_ret.rolling(rv_window).std()          # 截至 t 的过去 20 日
    fwd_H = log_ret.rolling(H).sum().shift(-H)       # t 时刻 = (t, t+H] 的累计

    df = pd.DataFrame({"r": log_ret, "rv": rv20, "fwd": fwd_H}).dropna()
    y = (df["fwd"] <= crash_threshold).astype(int).values

    n = len(df) - L + 1
    returns = np.stack([df["r"].values[i:i + L] for i in range(n)]).astype(np.float32)
    dates = df.index[L - 1:].to_numpy().astype("datetime64[D]")
    rv = df["rv"].values[L - 1:].astype(np.float32)
    return returns, dates, y[L - 1:], rv


# ═══════════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════════
def train_generator(returns, rv20, labels, cfg, epochs=None):
    """Phase 1 训练封装：返回 (eps_net, sampler)。"""
    device = torch.device(cfg.device)
    ds = TensorDataset(torch.tensor(returns, dtype=torch.float32),
                       torch.tensor(rv20, dtype=torch.float32),
                       torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    eps_net = EpsNet(cfg).to(device)
    opt = torch.optim.AdamW(eps_net.parameters(), lr=cfg.lr)
    sampler = DiffusionSampler(cfg, device)
    train_diffusion(eps_net, loader, opt, sampler, cfg, epochs=epochs)
    return eps_net, sampler


def oversample_crash(eps_net, sampler, X_crash, rv_crash, R, cfg):
    """暴跌类超额生成：每条真实暴跌窗口以自身 RV 与 σ 为条件生成 R 条。"""
    eps_net.eval()
    sig = X_crash.std(axis=1)
    X_synth = sampler.ddim_sample(
        eps_net, n=R * len(X_crash), y=1,
        rv=np.repeat(rv_crash, R),
        sigma_t=torch.tensor(np.repeat(sig, R), dtype=torch.float32),
    ).squeeze(1).cpu().numpy()                            # (R·N_crash, L)
    source_idx = np.repeat(np.arange(len(X_crash)), R)
    return X_synth, source_idx


def build_modes(returns, labels, dates, X_synth, source_idx,
                X_crash, d_crash, loo, d_synth, n_keep):
    """
    消融对照集构造。全部复用同一批合成候选，无需重训生成器：
      real      : 只用真实（含 scale_pos_weight ≈ 复制重加权的等价形式）
      replicate : 复制真实暴跌至与双边门控存活数相同——同扩增量的"复制 vs 生成"对照
                  （树模型下 ≈ real，正说明复制不注入新信息，故事的关键一行）
      full      : 全量合成（无门控）
      drop-near : 单边，只丢"太近"（防记忆）
      drop-far  : 单边，只丢"太远"（防幻觉）
      bilateral : 双边门控（本文方法）
    """
    lo, hi = np.quantile(loo, 0.05), np.quantile(loo, 0.95)
    d_synth_dates = d_crash[source_idx]
    modes = {"real": (returns, labels, dates)}
    if n_keep > 0:
        rng = np.random.default_rng(0)
        rep = rng.choice(len(X_crash), size=n_keep, replace=True)
        modes["replicate"] = (
            np.concatenate([returns, X_crash[rep]]),
            np.concatenate([labels, np.ones(n_keep)]),
            np.concatenate([dates, d_crash[rep]]))
    for name, m in {
        "full": np.ones(len(X_synth), dtype=bool),
        "drop-near": d_synth >= lo,
        "drop-far": d_synth <= hi,
        "bilateral": (d_synth >= lo) & (d_synth <= hi),
    }.items():
        if m.sum() > 0:
            modes[name] = (
                np.concatenate([returns, X_synth[m]]),
                np.concatenate([labels, np.ones(int(m.sum()))]),
                np.concatenate([dates, d_synth_dates[m]]))
    return modes


def perfold_robustness(returns, dates, labels, rv20, cfg,
                       model_kind="xgb", max_folds=3, epochs=150):
    """
    可选稳健性检验（默认关闭）：每个 fold 只用该 fold 的训练段重训生成器，
    防的是"生成器权重见过验证期数据"的分布级软泄漏。
    合成样本来源均在训练段内，天然满足 purge，无需额外处理。
    """
    out = []
    for tr, va, start in purged_expanding_walk_forward(
            len(returns), cfg.horizon, cfg.embargo_days, cfg.cv_min_train, cfg.cv_step):
        Xr, yr = returns[tr], labels[tr]
        if (yr == 1).sum() < 2 or (labels[va] == 1).sum() == 0:
            continue
        eps_net, sampler = train_generator(Xr, rv20[tr], yr, cfg, epochs=epochs)
        X_syn, src = oversample_crash(eps_net, sampler, Xr[yr == 1],
                                      rv20[tr][yr == 1], cfg.oversample_ratio, cfg)
        loo, d = dcr_distances(X_syn, Xr[yr == 1])
        mask, _ = gate_from_distances(d, loo, cfg.dcr_lo_q, cfg.dcr_hi_q)
        Xk = X_syn[mask]
        X_aug = np.concatenate([Xr, Xk]) if len(Xk) else Xr
        y_aug = np.concatenate([yr, np.ones(len(Xk))]) if len(Xk) else yr
        clf = make_clf(model_kind, (y_aug == 0).sum() / max((y_aug == 1).sum(), 1))
        clf.fit(X_aug, y_aug)
        p = clf.predict_proba(returns[va])[:, 1]
        yhat = (p > 0.5).astype(int)
        yv = labels[va]
        out.append({"start": str(dates[start]),
                    "Recall": recall_score(yv, yhat, zero_division=0),
                    "PR-AUC": average_precision_score(yv, p),
                    "F1": f1_score(yv, yhat, zero_division=0)})
        print(f"  [perfold] fold@{dates[start]}  {out[-1]}")
        if len(out) >= max_folds:
            break
    if not out:
        print("  [perfold] 无可用 fold")
    return out


def run_pipeline(returns, dates, labels, rv20, cfg=None):
    """
    returns: (N, L) 32 天日对数收益窗口
    dates:   (N,)  窗口结束日期（numpy datetime64，升序）
    labels:  (N,)  二分类暴跌标签
    rv20:    (N,)  已实现波动率
    """
    cfg = cfg or Config()
    device = torch.device(cfg.device)
    print(f"[Init] device={device}, N={len(returns)}, "
          f"crash={int((labels == 1).sum())}")

    # ---- Phase 1: 训练生成器 + 暴跌类超额生成 ----
    print("[Phase 1] training diffusion ...")
    eps_net, sampler = train_generator(returns, rv20, labels, cfg)
    crash_mask = labels == 1
    X_crash, rv_crash, d_crash = returns[crash_mask], rv20[crash_mask], dates[crash_mask]
    assert len(X_crash) >= 2, "真实暴跌样本 < 2，DCR 基线无法计算"
    R = cfg.oversample_ratio
    print(f"[Phase 1] oversampling x{R} for crash class ...")
    X_synth, source_idx = oversample_crash(eps_net, sampler, X_crash, rv_crash, R, cfg)

    # ---- Phase 2: DCR 双边门控（含敏感性扫描与 bootstrap CI）----
    loo, d_synth = dcr_distances(X_synth, X_crash)
    lo_thr, hi_thr = np.quantile(loo, cfg.dcr_lo_q), np.quantile(loo, cfg.dcr_hi_q)
    print(f"[Phase 2] 甜蜜区 [{lo_thr:.4f}, {hi_thr:.4f}]")
    if len(loo) >= 5:   # 暴跌样本少时分位数不稳定，bootstrap CI 可视化其宽度
        rng = np.random.default_rng(0)
        boots = np.array([[np.quantile(rng.choice(loo, len(loo)), q)
                           for q in (cfg.dcr_lo_q, cfg.dcr_hi_q)]
                          for _ in range(500)])
        print(f"[Phase 2] 阈值 bootstrap 95%CI: "
              f"下限{np.percentile(boots[:, 0], [2.5, 97.5]).round(4)}, "
              f"上限{np.percentile(boots[:, 1], [2.5, 97.5]).round(4)}")
    print("[Phase 2] 分位数敏感性（λ_eff 涌现的稳健性）:")
    for ql, qh in ((0.01, 0.99), (0.05, 0.95), (0.10, 0.90)):
        m = (d_synth >= np.quantile(loo, ql)) & (d_synth <= np.quantile(loo, qh))
        print(f"  Q{ql:.0%}/Q{qh:.0%}: keep={int(m.sum())}, "
              f"λ_eff={m.sum() / len(X_crash):.2f}")
    mask, lam_eff = gate_from_distances(d_synth, loo, cfg.dcr_lo_q, cfg.dcr_hi_q)
    X_kept, src_kept = X_synth[mask], source_idx[mask]
    print(f"[Phase 2] {len(X_synth)} → {len(X_kept)} survived, "
          f"λ_eff = {lam_eff:.2f} (自动决定)")

    # ---- Phase 3: 程式化事实检验 ----
    rep = None
    if len(X_kept) > 0:
        rep = stylized_facts_report(X_crash.ravel(), X_kept.ravel())
        print(f"[Phase 3] crash类: K_real={rep['excess_kurtosis_real']:.2f} "
              f"K_synth={rep['excess_kurtosis_synth']:.2f} "
              f"W1={rep['wasserstein_1']:.4f} "
              f"decay_L1={rep['vol_decay_l1']:.4f}")
    else:
        print("[Phase 3] 警告：门控后无存活样本（考虑增大 R 或放宽分位数）")
    # y=0 质检：证明生成器学到整体动力学而非仅暴跌形态（只统计，不进下游）
    nc = np.where(labels == 0)[0]
    if len(nc) > 0:
        sub = np.random.default_rng(1).choice(nc, size=min(len(nc), 1000), replace=False)
        X0 = sampler.ddim_sample(
            eps_net, n=len(sub), y=0, rv=rv20[sub],
            sigma_t=torch.tensor(returns[sub].std(axis=1), dtype=torch.float32),
        ).squeeze(1).cpu().numpy()
        rep0 = stylized_facts_report(returns[sub].ravel(), X0.ravel())
        print(f"[Phase 3] 非暴跌类: K_real={rep0['excess_kurtosis_real']:.2f} "
              f"K_synth={rep0['excess_kurtosis_synth']:.2f} "
              f"W1={rep0['wasserstein_1']:.4f} "
              f"decay_L1={rep0['vol_decay_l1']:.4f}")

    # ---- Phase 4: 主结果（三模型）+ 消融（xgb）----
    modes = build_modes(returns, labels, dates, X_synth, source_idx,
                        X_crash, d_crash, loo, d_synth, len(X_kept))
    results = {}
    if "bilateral" in modes:
        Xa, ya, da = modes["bilateral"]
        print("[Phase 4] purged walk-forward CV (bilateral, 3 models) ...")
        for kind in ("xgb", "lgbm", "mlp"):
            try:
                m = train_and_evaluate(
                    Xa, ya, da, n_real=len(returns),
                    H=cfg.horizon, embargo=cfg.embargo_days,
                    min_train=cfg.cv_min_train, step=cfg.cv_step,
                    model_kind=kind)
            except ImportError as e:
                print(f"[Phase 4/{kind}] 跳过（{e}）")
                continue
            if m is not None:
                results[f"bilateral/{kind}"] = m
                print(f"[Phase 4/bilateral-{kind}] {m}")
    print("[Phase 4] ablation (xgb) ...")
    for name, (Xa, ya, da) in modes.items():
        if name == "bilateral" and "bilateral/xgb" in results:
            m = results["bilateral/xgb"]        # 已跑过，复用
        else:
            m = train_and_evaluate(
                Xa, ya, da, n_real=len(returns),
                H=cfg.horizon, embargo=cfg.embargo_days,
                min_train=cfg.cv_min_train, step=cfg.cv_step,
                model_kind="xgb")
        if m is not None:
            results[f"ablation/{name}"] = m
            print(f"[Phase 4/ablation-{name}] {m}")
    return results, rep, lam_eff


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # ── 方式 A：真实数据（接好数据后取消注释，并删掉方式 B）──
    # df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
    # returns, dates, labels, rv20 = make_dataset(df["close"])
    # cfg = Config()                                       # 正式跑：默认 epochs=500
    # results, rep, lam_eff = run_pipeline(returns, dates, labels, rv20, cfg)
    # perfold_robustness(returns, dates, labels, rv20, cfg)  # 可选：软泄漏稳健性检验

    # ── 方式 B：烟雾测试（随机数据，只验证代码跑得通，结果无意义）──
    SMOKE_TEST = True
    if SMOKE_TEST:
        rng = np.random.default_rng(0)
        N = 1500
        returns = rng.normal(0, 0.02, (N, Config.L)).astype(np.float32)
        dates = pd.date_range("2010-01-01", periods=N, freq="B") \
                    .to_numpy().astype("datetime64[D]")
        labels = (rng.random(N) < 0.08).astype(int)      # 8% 暴跌率
        rv20 = returns.std(axis=1) + 1e-4
        run_pipeline(returns, dates, labels, rv20,
                     cfg=Config(epochs=30, cv_step=120))  # 烟雾测试用小 epoch / 大步长