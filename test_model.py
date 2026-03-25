import torch
from models.unet import UNet

model = UNet()

x = torch.randn(2, 2, 256, 256)  # batch = 2

y = model(x)

print("Output shape:", y.shape)