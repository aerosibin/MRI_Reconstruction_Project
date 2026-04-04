import numpy as np
from simulation.mask import variable_density_mask, multi_layer_variable_density_mask


def cartesian_undersample_kspace(
    stack, accel=4, center_frac=0.08, seed=None, layers=None
):
    """Undersample a cine stack in k-space per-frame using a simple
    variable-density Cartesian mask.

    stack: (T,H,W) real images (magnitude). Returns kspace_under (T,H,W) complex
    and mask (H,W) boolean applied to all frames.
    """
    T, H, W = stack.shape
    rng = np.random.RandomState(seed)

    # build a single mask applied to all frames for simplicity
    if layers is not None:
        mask = multi_layer_variable_density_mask((H, W), layers=layers, seed=seed)
    else:
        mask = np.zeros((H, W), dtype=bool)
        center_w = int(W * center_frac)
        start = W // 2 - center_w // 2
        end = start + center_w
        mask[:, start:end] = True

        prob = (W / float(accel) - center_w) / max(1, (W - center_w))
        for i in range(W):
            if i < start or i >= end:
                if rng.rand() < prob:
                    mask[:, i] = True

    # compute k-space for each frame and apply mask
    k_under = np.zeros((T, H, W), dtype=np.complex64)
    for t in range(T):
        im = stack[t]
        k = np.fft.fftshift(np.fft.fft2(im))
        k_under[t] = k * mask

    return k_under, mask
