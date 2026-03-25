import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import numpy as np


# =========================
# MSE Loss
# =========================
mse_loss = nn.MSELoss()


# =========================
# SSIM Loss
# =========================
def ssim_loss(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    ssim_total = 0

    for i in range(pred.shape[0]):
        ssim_val = ssim(
            target[i, 0],
            pred[i, 0],
            data_range=1.0
        )
        ssim_total += ssim_val

    return 1 - (ssim_total / pred.shape[0])


# =========================
# Combined Loss
# =========================
def combined_loss(pred, target, alpha=0.8):
    mse = mse_loss(pred, target)
    ssim_l = ssim_loss(pred, target)

    return alpha * mse + (1 - alpha) * torch.tensor(ssim_l)