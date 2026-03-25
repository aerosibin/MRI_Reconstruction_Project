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

        # Traverse folders
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)

            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0  # normalize

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