"""
Phase 1: Lightweight Conditional Diffusion Engine.
Components:
- 1D Dilated Causal TCN (RF=73 >= L=32, zero future lookahead)
- FiLM (Feature-wise Linear Modulation) conditioning on RV20 and crash label
- Local Volatility Scale Normalization / De-normalization (x_tilde = r / sigma_t)
- 20-step Fast Deterministic DDIM Sampler (eta=0, ODE trajectory)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from src.config import TailDiffConfig


class CausalConv1d(nn.Module):
    """严格因果一维卷积：左侧填充 (k-1)*dilation，杜绝未来信息穿越。"""
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        return self.conv(F.pad(x, (self.pad, 0)))


class FiLM(nn.Module):
    """特征线性调制层：h' = (1 + gamma(c)) * h + beta(c)"""
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # h: (B, C, L), c: (B, cond_dim)
        g = self.to_gamma(c).unsqueeze(-1)
        b = self.to_beta(c).unsqueeze(-1)
        return (1.0 + g) * h + b


class ResDilatedBlock(nn.Module):
    """因果膨胀残差块：CausalConv -> GroupNorm -> FiLM -> GELU x2 + Residual"""
    def __init__(self, C: int, k: int, d: int, gn_groups: int, cond_dim: int):
        super().__init__()
        self.conv1 = CausalConv1d(C, C, k, d)
        self.conv2 = CausalConv1d(C, C, k, d)
        self.norm1 = nn.GroupNorm(gn_groups, C)
        self.norm2 = nn.GroupNorm(gn_groups, C)
        self.film1 = FiLM(cond_dim, C)
        self.film2 = FiLM(cond_dim, C)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.film1(self.norm1(self.conv1(x)), c))
        h = self.film2(self.norm2(self.conv2(h)), c)
        return F.gelu(x + h)


class EpsNet(nn.Module):
    """轻量级去噪网络 (参数量 ~ 0.38M)"""
    def __init__(self, cfg: TailDiffConfig):
        super().__init__()
        C = cfg.channels
        self.cfg = cfg
        self.t_emb = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        self.y_emb = nn.Embedding(2, 32)
        cond_dim = 64 + 1 + 32  # timestep_emb + log(RV20) + label_emb

        self.in_proj = CausalConv1d(1, C, 1, 1)
        self.blocks = nn.ModuleList([
            ResDilatedBlock(C, cfg.kernel_size, d, cfg.num_groups, cond_dim)
            for d in cfg.dilations
        ])
        self.out_proj = nn.Conv1d(C, 1, 1)

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, rv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        t_embed = self.t_emb(self._sinusoidal_embedding(t, dim=64))
        rv_feat = torch.log(rv.clamp(min=1e-6)).view(B, 1)
        y_feat = self.y_emb(y.long())
        c = torch.cat([t_embed, rv_feat, y_feat], dim=-1)

        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, c)
        return self.out_proj(h)


class DiffusionSampler:
    """DDPM 训练前向加噪 + 20 步确定性 DDIM 采样器"""
    def __init__(self, cfg: TailDiffConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_diffusion_steps)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    @staticmethod
    def _to_tensor(v, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(v):
            v = torch.tensor(np.asarray(v, dtype=np.float32))
        return v.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def ddim_sample(self, eps_net: nn.Module, n: int, y: int, rv: np.ndarray, sigma_t: np.ndarray, S: int = None) -> torch.Tensor:
        """
        确定性 ODE DDIM 采样 (eta=0)，耗时仅 ~12ms。
        生成标准化形态 x_tilde 并经 sigma_t 反归一化还原。
        """
        S = S or self.cfg.ddim_steps
        taus = torch.linspace(self.cfg.num_diffusion_steps - 1, 0, S, device=self.device).long()
        x = torch.randn(n, 1, self.cfg.L, device=self.device)
        y_ten = torch.full((n,), int(y), device=self.device, dtype=torch.long)
        rv_ten = self._to_tensor(rv, self.device)
        if rv_ten.dim() == 0:
            rv_ten = rv_ten.expand(n)
        sigma_ten = self._to_tensor(sigma_t, self.device)

        for i in range(S):
            t = taus[i]
            ab_t = self.alpha_bar[t].view(-1, 1, 1)
            eps = eps_net(x, t.expand(n), rv_ten, y_ten)
            x0_pred = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()

            if i < S - 1:
                ab_next = self.alpha_bar[taus[i + 1]].view(-1, 1, 1)
                x = ab_next.sqrt() * x0_pred + (1.0 - ab_next).sqrt() * eps
            else:
                x = x0_pred

        # 局部波动率反归一化: x_synth = x_tilde * sigma_t
        return x * sigma_ten.view(-1, 1, 1)


def train_diffusion(eps_net: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                    sampler: DiffusionSampler, cfg: TailDiffConfig, epochs: int = None):
    """局部波动率归一化训练: x_tilde = r / sigma_t"""
    epochs = epochs or cfg.epochs
    eps_net.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (r, rv, y) in loader:
            r = r.to(cfg.device).unsqueeze(1)  # (B, 1, L)
            rv, y = rv.to(cfg.device), y.to(cfg.device)
            sigma = r.std(dim=-1, keepdim=True).clamp(min=1e-4)  # (B, 1, 1)
            x_tilde = r / sigma

            t = torch.randint(0, cfg.num_diffusion_steps, (r.size(0),), device=cfg.device)
            noise = torch.randn_like(x_tilde)
            x_noisy = sampler.q_sample(x_tilde, t, noise)
            
            eps_pred = eps_net(x_noisy, t, rv, y)
            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"  [Diffusion Engine] Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(loader):.5f}")
