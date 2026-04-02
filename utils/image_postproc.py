import numpy as np
import cv2


def unsharp_mask(img, kernel_size=5, sigma=1.0, strength=1.0):
    """Apply unsharp mask to a single-channel float image in [0,1]."""
    img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype("uint8")
    blurred = cv2.GaussianBlur(img_u8, (kernel_size, kernel_size), sigma)
    sharp = cv2.addWeighted(img_u8, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharp.astype("float32") / 255.0, 0.0, 1.0)
