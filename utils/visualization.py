import os
import numpy as np
import cv2


def make_montage(gt_paths, zf_paths, pred_paths, thumb_size=128, cols=5):
    """Create a montage image with rows [GT | ZF | PRED] for each sample.

    Returns normalized float image in [0,1].
    """
    imgs = []
    for gt, zf, pred in zip(gt_paths, zf_paths, pred_paths):
        g = cv2.imread(gt, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
        z = (
            (cv2.imread(zf, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0)
            if (zf and os.path.exists(zf))
            else np.zeros_like(g)
        )
        p = cv2.imread(pred, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0

        # Resize to thumbnail
        g = cv2.resize(g, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        z = cv2.resize(z, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        p = cv2.resize(p, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)

        # Stack horizontally for this sample
        row = np.hstack([g, z, p])
        imgs.append(row)

    # arrange into grid
    rows = []
    for i in range(0, len(imgs), cols):
        block = imgs[i : i + cols]
        # pad if last row shorter
        while len(block) < cols:
            block.append(np.zeros_like(block[0]))
        rows.append(np.hstack(block))

    montage = np.vstack(rows)
    montage = np.clip(montage, 0.0, 1.0)
    return montage
