import argparse
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


def train(
    train_dir="data/processed/train",
    batch_size=8,
    epochs=50,
    lr=1e-4,
    val_size=400,
    clip_norm=1.0,
    patience=10,
    num_workers=8,
    use_amp=False,
    device=None,
    save_prefix="model_recon",
):
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Print configuration for reproducibility/diagnostics
    print(f"Training on device: {device}")
    print(
        f"train_dir={train_dir} batch_size={batch_size} epochs={epochs} lr={lr} val_size={val_size} num_workers={num_workers} use_amp={use_amp} save_prefix={save_prefix}"
    )

    # Dataset and split
    dataset = MRIDataset(train_dir)
    train_ds, val_ds = _make_val_split(dataset, val_size=val_size)

    pin_memory = True if device.type == "cuda" else False
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Model
    model = UNet().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler and training params
    # Scheduler will reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    best_val = float("inf")
    no_improve = 0

    os.makedirs("outputs", exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for x, y in loop:
            x, y = (
                x.to(device, non_blocking=pin_memory),
                y.to(device, non_blocking=pin_memory),
            )

            optimizer.zero_grad()

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pred = model(x)
                    loss = combined_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = combined_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            total_loss += float(loss.item())
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = (
                    x.to(device, non_blocking=pin_memory),
                    y.to(device, non_blocking=pin_memory),
                )
                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        pred = model(x)
                        l = combined_loss(pred, y)
                else:
                    pred = model(x)
                    l = combined_loss(pred, y)
                val_loss += float(l.item())

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
            ckpt = f"{save_prefix}_best.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"Saved best model -> {ckpt}")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    # Also save final model
    final_ckpt = f"{save_prefix}_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final model -> {final_ckpt}")


def main():
    parser = argparse.ArgumentParser(
        description="Train UNet reconstruction (GPU-ready)"
    )
    parser.add_argument("--train-dir", default="data/processed/train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--use-amp", action="store_true", help="use CUDA AMP (requires cuda)"
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="model_recon",
        help="checkpoint prefix at repo root",
    )
    args = parser.parse_args()

    # Set cudnn benchmark for speed when using fixed-size inputs and GPU
    if args.device is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True

    train(
        train_dir=args.train_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_size=args.val_size,
        clip_norm=args.clip_norm,
        patience=args.patience,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        device=dev,
        save_prefix=args.save_prefix,
    )


if __name__ == "__main__":
    main()
