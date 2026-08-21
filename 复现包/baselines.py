"""
Baseline Implementations for Financial Tail-Risk Benchmark:
1. No-Aug (Original Imbalanced)
2. Class-Weighted / Cost-Sensitive Tuning
3. Volatility Rule (RV20 > threshold)
4. SMOTE (Linear Geometric Interpolation)
5. Borderline-SMOTE
6. Conditional VAE (C-VAE)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional


class SMOTEAugmenter:
    """SMOTE (Synthetic Minority Over-sampling Technique) 一维时序特征插值。"""
    def __init__(self, k_neighbors: int = 5, seed: int = 42):
        self.k = k_neighbors
        self.seed = seed

    def fit_sample(self, X_minority: np.ndarray, n_samples: int) -> np.ndarray:
        if len(X_minority) <= 1 or n_samples <= 0:
            return np.empty((0, X_minority.shape[1]), dtype=np.float32)
        
        k = min(self.k, len(X_minority) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_minority)
        _, indices = nbrs.kneighbors(X_minority)
        
        rng = np.random.default_rng(self.seed)
        synthetic = []
        for _ in range(n_samples):
            i = rng.integers(0, len(X_minority))
            nn_idx = rng.choice(indices[i, 1:])  # 排除自身
            diff = X_minority[nn_idx] - X_minority[i]
            gap = rng.random()
            synthetic.append(X_minority[i] + gap * diff)
            
        return np.array(synthetic, dtype=np.float32)


class BorderlineSMOTEAugmenter:
    """Borderline-SMOTE: 仅在决策边界（DANGER 区域）的少数类样本周围进行插值。"""
    def __init__(self, k_neighbors: int = 5, seed: int = 42):
        self.k = k_neighbors
        self.seed = seed

    def fit_sample(self, X: np.ndarray, y: np.ndarray, n_samples: int) -> np.ndarray:
        minority_idx = np.where(y == 1)[0]
        if len(minority_idx) <= 1 or n_samples <= 0:
            return np.empty((0, X.shape[1]), dtype=np.float32)
        
        k = min(self.k, len(X) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
        _, indices = nbrs.kneighbors(X[minority_idx])
        
        # 寻找处于边界危险区的少数类样本
        danger_indices = []
        for row_idx, nn_list in enumerate(indices):
            # 统计近邻中多数类的数量
            maj_count = np.sum(y[nn_list[1:]] == 0)
            if k / 2 <= maj_count < k:  # 属于 DANGER 区域
                danger_indices.append(minority_idx[row_idx])
                
        if not danger_indices:
            danger_indices = list(minority_idx)
            
        X_danger = X[danger_indices]
        smote = SMOTEAugmenter(k_neighbors=min(self.k, len(X_danger) - 1), seed=self.seed)
        return smote.fit_sample(X_danger, n_samples)


class CVAEGenerator(nn.Module):
    """轻量级条件变分自编码器 (C-VAE) 基线。"""
    def __init__(self, feat_dim: int = 32, cond_dim: int = 1, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        # Encoder: (x, c) -> (mu, logvar)
        self.enc = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: (z, c) -> x_recon
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feat_dim)
        )
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, c: torch.Tensor, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z, c)
