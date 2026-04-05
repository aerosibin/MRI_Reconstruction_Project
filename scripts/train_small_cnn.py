import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import joblib


class ImageFolderDataset(Dataset):
    def __init__(self, root, size=(64, 64)):
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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, self.size, interpolation=cv2.INTER_AREA)
        im = im.astype("float32") / 255.0
        x = im.reshape(1, self.size[0], self.size[1])
        y = self.labels[idx]
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, nclass)

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


def train(root_train, root_val, out_dir, epochs=3, batch_size=32, size=(64, 64)):
    ds = ImageFolderDataset(root_train, size=size)
    val_ds = ImageFolderDataset(root_val, size=size)
    classes = ds.classes
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = "cpu"
    net = SmallCNN(len(classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        net.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = net(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {ep} done")

    os.makedirs(out_dir, exist_ok=True)
    # save state_dict for model_wrapper
    torch.save({"state_dict": net.state_dict()}, os.path.join(out_dir, "small_cnn.pth"))
    # also save classes
    joblib.dump({"classes": classes}, os.path.join(out_dir, "small_cnn_meta.joblib"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()
    train(args.train, args.val, args.out)
