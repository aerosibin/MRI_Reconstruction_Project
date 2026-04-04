import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Implement a small, differentiable SSIM in PyTorch to support backprop on CPU/GPU
def _gaussian_window(window_size, sigma, channel):
    coords = torch.arange(window_size).float() - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.repeat(channel, 1, 1, 1)
    return window


# =========================
# MSE Loss
# =========================
mse_loss = nn.MSELoss()


# =========================
# SSIM Loss
# =========================
def ssim_loss(pred, target, window_size=11, sigma=1.5, K=(0.01, 0.03)):
    # pred, target: tensors with shape (N, 1, H, W), values in [0,1]
    # returns: 1 - mean SSIM over batch (differentiable)
    device = pred.device
    channel = pred.size(1)
    window = _gaussian_window(window_size, sigma, channel).to(device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(target * target, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(pred * target, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = (K[0] * 1.0) ** 2
    C2 = (K[1] * 1.0) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return 1.0 - ssim_map.mean()


# =========================
# Combined Loss
# =========================
def combined_loss(pred, target, alpha=0.8):
    mse = mse_loss(pred, target)
    ssim_l = ssim_loss(pred, target)

    # ensure ssim_l is a tensor on the same device
    if not isinstance(ssim_l, torch.Tensor):
        ssim_l = torch.tensor(ssim_l, device=pred.device)

    return alpha * mse + (1.0 - alpha) * ssim_l
