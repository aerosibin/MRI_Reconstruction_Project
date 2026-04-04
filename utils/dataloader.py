import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# =========================
# Helper: Create Mask
# =========================
def create_mask(shape, center_fraction=0.08, acceleration=4):
    H, W = shape

    mask = np.zeros((H, W))

    # Fully sampled center
    center_width = int(W * center_fraction)
    center_start = W // 2 - center_width // 2
    center_end = center_start + center_width

    mask[:, center_start:center_end] = 1

    # Random sampling outside center
    prob = (W / acceleration - center_width) / (W - center_width)

    for i in range(W):
        if i < center_start or i > center_end:
            if np.random.rand() < prob:
                mask[:, i] = 1

    return mask


# =========================
# Dataset Class
# =========================
class MRIDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []

        # Traverse folders or files. Support two structures:
        # 1) root_dir/class_name/*.png (class subfolders)
        # 2) root_dir/*.png (images directly)
        entries = sorted(os.listdir(root_dir))
        for name in entries:
            path = os.path.join(root_dir, name)
            if os.path.isdir(path):
                for img_name in sorted(os.listdir(path)):
                    img_path = os.path.join(path, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
            elif os.path.isfile(path):
                # root_dir contains images directly
                self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0  # normalize

        # --- Simple augmentations (light) ---
        # Random flips
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
        if np.random.rand() < 0.5:
            img = np.flipud(img)

        # Small random rotation (-10 to 10 degrees)
        angle = (np.random.rand() * 20.0) - 10.0
        if abs(angle) > 0.5:
            H, W = img.shape
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            img = cv2.warpAffine(
                img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

        # Intensity jitter
        if np.random.rand() < 0.3:
            jitter = 0.9 + 0.2 * (np.random.rand() - 0.5)
            img = np.clip(img * jitter, 0.0, 1.0)

        # Ensure contiguous array (fix negative-stride views from flips/rotations)
        img = np.ascontiguousarray(img)

        # FFT → k-space
        kspace = np.fft.fft2(img)
        kspace = np.fft.fftshift(kspace)

        # Mask
        mask = create_mask(img.shape)
        kspace_under = kspace * mask

        # IFFT → zero-filled reconstruction
        img_under = np.fft.ifftshift(kspace_under)
        img_under = np.fft.ifft2(img_under)

        # Split real + imaginary
        input_img = np.stack([img_under.real, img_under.imag], axis=0)

        # Ground truth (magnitude)
        target = img

        # Convert to tensor
        input_img = torch.tensor(input_img, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        return input_img, target
