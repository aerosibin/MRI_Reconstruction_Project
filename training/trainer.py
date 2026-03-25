import torch
from torch.utils.data import DataLoader
from utils.dataloader import MRIDataset
from models.unet import UNet
from training.loss import combined_loss
from tqdm import tqdm


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = MRIDataset("data/processed/train")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model
    model = UNet().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(loader)

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            # Forward
            pred = model(x)

            # Loss
            loss = combined_loss(pred, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader)}")

    # Save model
    torch.save(model.state_dict(), "outputs/model.pth")