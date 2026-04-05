import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import joblib
from collections import Counter


class ImageFolderDataset(Dataset):
    def __init__(self, root, size=(64, 64), augment=False):
        self.paths = []
        self.labels = []
        self.classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
        for cls in self.classes:
            p = os.path.join(root, cls)
            for f in os.listdir(p):
                if f.lower().endswith(".png"):
                    self.paths.append(os.path.join(p, f))
                    self.labels.append(self.classes.index(cls))
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _augment(self, im):
        if np.random.rand() < 0.5:
            im = np.fliplr(im)
        if np.random.rand() < 0.3:
            im = np.clip(im * (0.9 + 0.2 * np.random.rand()), 0.0, 1.0)
        return im

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, self.size, interpolation=cv2.INTER_AREA)
        im = im.astype("float32") / 255.0
        if self.augment:
            im = self._augment(im)
        # make contiguous to avoid negative-stride numpy arrays from flips
        im = np.ascontiguousarray(im)
        x = im.reshape(1, self.size[0], self.size[1])
        y = self.labels[idx]
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128, nclass)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(root_train, root_val, out_dir, epochs=10, batch_size=32, size=(64, 64)):
    ds = ImageFolderDataset(root_train, size=size, augment=True)
    val_ds = ImageFolderDataset(root_val, size=size, augment=False)
    classes = ds.classes
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = "cpu"
    net = SmallCNN(len(classes)).to(device)
    # compute class weights
    counts = Counter(ds.labels)
    weights = [0.0] * len(classes)
    for i, c in counts.items():
        weights[i] = 1.0 / (c + 1e-9)
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = net(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / max(1, len(train_loader))
        print(f"Epoch {ep} train_loss={avg:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    # save state_dict and meta
    torch.save(
        {"state_dict": net.state_dict()}, os.path.join(out_dir, "small_cnn_adv.pth")
    )
    joblib.dump(
        {"classes": classes}, os.path.join(out_dir, "small_cnn_adv_meta.joblib")
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.train, args.val, args.out, epochs=args.epochs)
