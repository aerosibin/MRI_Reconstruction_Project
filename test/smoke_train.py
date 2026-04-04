import torch
from torch.utils.data import DataLoader, random_split
from utils.dataloader import MRIDataset
from models.unet import UNet
from training.loss import combined_loss


def run():
    root = "data/processed/train"
    ds = MRIDataset(root)
    n = len(ds)
    if n < 2:
        print("Not enough samples for smoke test")
        return
    val = min(10, max(1, n // 10))
    train_n = min(20, n - val)
    if train_n <= 0 or train_n + val != n:
        print("Not enough samples for smoke test")
        return
    train_ds, val_ds = random_split(ds, [train_n, val])
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    device = torch.device("cpu")
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = combined_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print("train step loss", loss.item())

    model.eval()
    with torch.no_grad():
        vsum = 0.0
        count = 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            l = combined_loss(pred, yb)
            vsum += l.item()
            count += 1
        print("val loss", vsum / max(1, count))

    torch.save(model.state_dict(), "model_smoke.pth")
    print("Wrote model_smoke.pth")


if __name__ == "__main__":
    run()
