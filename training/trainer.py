import torch
from torch.utils.data import DataLoader, random_split
from utils.dataloader import MRIDataset
from models.unet import UNet
from training.loss import combined_loss
from tqdm import tqdm
import os


def _make_val_split(dataset, val_size=200):
    # Create a deterministic small validation split
    n = len(dataset)
    val = min(val_size, int(0.2 * n))
    train = n - val
    return random_split(dataset, [train, val])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and split
    dataset = MRIDataset("data/processed/train")
    train_ds, val_ds = _make_val_split(dataset, val_size=200)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # Model
    model = UNet().to(device)

    # Optimizer (reduced LR for stability)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Scheduler and training params
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    epochs = 10
    clip_norm = 1.0
    best_val = float("inf")
    patience = 5
    no_improve = 0

    os.makedirs("outputs", exist_ok=True)

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
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(loader)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                l = combined_loss(pred, y)
                val_loss += l.item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float("nan")

        print(
            f"Epoch {epoch + 1} Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}"
        )

        # Scheduler step
        if not (val_loss != val_loss):
            scheduler.step(val_loss)

        # Early stopping & checkpoint
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            # Save best model at repository root (do not place model.pth under outputs/)
            torch.save(model.state_dict(), "model_best.pth")
            print("Saved best model -> model_best.pth")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    # Also save final model (not best) at repo root for reproducibility
    torch.save(model.state_dict(), "model_final.pth")
