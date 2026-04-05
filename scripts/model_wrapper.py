import torch
import torch.nn.functional as F
import numpy as np
import os


class TorchWrapper:
    """Lightweight wrapper that exposes sklearn-like decision_function and predict_proba
    by loading a saved torch model file on first call. Stores class order externally.
    """

    def __init__(self, model_path, classes, device="cpu"):
        self.model_path = model_path
        self.classes = list(classes)
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import importlib

            # model class defined in training script; we simply load state_dict into a small net
            # define small net architecture here to match training
            import torch.nn as nn

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

            nclass = len(self.classes)
            net = SmallCNN(nclass)
            state = torch.load(self.model_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                net.load_state_dict(state["state_dict"])
            else:
                net.load_state_dict(state)
            net.to(self.device)
            net.eval()
            self._model = net

    def decision_function(self, X):
        """X: numpy array shape (N, D) where D = H*W (grayscale)
        returns logits numpy array shape (N, nclass)
        """
        self._load()
        import torch

        N = X.shape[0]
        H = W = int(np.sqrt(X.shape[1]))
        x = X.reshape(N, 1, H, W).astype("float32")
        t = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logits = self._model(t).cpu().numpy()
        return logits

    def predict_proba(self, X):
        logits = self.decision_function(X)
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
