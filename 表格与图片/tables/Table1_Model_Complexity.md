# Table 1. Computational Complexity and Latency Benchmark across Architectures

| Architecture | Model Parameters | FLOPs (M) | Sampling Latency (ms) | Peak Memory (MB) | Edge Deployable |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard 2D U-Net (DDPM-1000)** | 12.40 M | 1450.0 | 420.0 ms | 1,850 MB | No (GPU Server Only) |
| **TimeGAN (Autoregressive GAN)** | 1.80 M | 124.0 | 45.0 ms | 320 MB | Marginal (Mode Collapse) |
| **CSDI (Transformer Diffusion)** | 4.20 M | 380.0 | 185.0 ms | 680 MB | GPU Required |
| **TailDiff (1D FiLM Conv, FP32, Ours)** | **0.91 M** | **24.2** | **5.09 ms (DDIM-20)** | **64 MB** | **Yes (Real-time CPU)** |
| **TailDiff (8-bit Quantized, Ours)** | **0.91 M** | **24.2** | **1.85 ms (DDIM-20)** | **18 MB** | **Yes (Ultra-Low Latency)** |
