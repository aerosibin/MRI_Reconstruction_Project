import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Ensure repo root is on python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.unet import UNet
from utils.dataloader import MRIDataset
import numpy as np
import cv2
from skimage.metrics import (
    peak_signal_noise_ratio as psnr_sk,
    structural_similarity as ssim_sk,
)


def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
        except Exception:
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
                model.load_state_dict(sd)
    else:
        model = ckpt
    return model


def to_np(x):
    return x.detach().cpu().numpy()


def run_on_folder(folder, model, device, out_dir, limit=None):
    ds = MRIDataset(folder)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    records = []
    for idx, (x, y) in enumerate(loader):
        if limit is not None and idx >= limit:
            break

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            pred = model(x)

        pred_np = to_np(pred[:, 0, :, :])[0]
        gt_np = to_np(y[:, 0, :, :])[0]
        inp = to_np(x)[0]
        zf = np.sqrt(inp[0] ** 2 + inp[1] ** 2)

        pred_img = np.clip(pred_np, 0.0, 1.0)
        gt_img = np.clip(gt_np, 0.0, 1.0)
        zf_img = np.clip(zf, 0.0, 1.0)

        try:
            p = psnr_sk(gt_img, pred_img, data_range=1.0)
        except Exception:
            p = float("nan")
        try:
            s = ssim_sk(gt_img, pred_img, data_range=1.0)
        except Exception:
            s = float("nan")

        records.append({"idx": idx, "psnr": float(p), "ssim": float(s)})

        cv2.imwrite(
            os.path.join(out_dir, f"{idx:04d}_gt.png"), (gt_img * 255).astype("uint8")
        )
        cv2.imwrite(
            os.path.join(out_dir, f"{idx:04d}_zf.png"), (zf_img * 255).astype("uint8")
        )
        cv2.imwrite(
            os.path.join(out_dir, f"{idx:04d}_pred.png"),
            (pred_img * 255).astype("uint8"),
        )

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="data/processed/test")
    parser.add_argument("--out", type=str, default="outputs/inference_baseline")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="model.pth")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = UNet().to(device)
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found at {args.ckpt}")
        return
    model = load_checkpoint(args.ckpt, model, device)
    model.eval()

    sequences = [
        os.path.join(args.src, p)
        for p in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, p))
    ]
    if not sequences:
        sequences = [args.src]

    aggregate = []
    for seq in sequences:
        name = os.path.basename(seq.rstrip(os.sep))
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        recs = run_on_folder(seq, model, device, out_dir, limit=args.limit)
        aggregate.extend([{"sequence": name, **r} for r in recs])

    if aggregate:
        import csv

        out_metrics = os.path.join(args.out, "metrics_aggregate.csv")
        keys = list(aggregate[0].keys())
        with open(out_metrics, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in aggregate:
                writer.writerow(r)

        ps = [r["psnr"] for r in aggregate if not np.isnan(r["psnr"])]
        ss = [r["ssim"] for r in aggregate if not np.isnan(r["ssim"])]
        if ps:
            print(
                f"Processed {len(aggregate)} images. PSNR mean={np.mean(ps):.3f} std={np.std(ps):.3f}"
            )
        if ss:
            print(f"SSIM mean={np.mean(ss):.4f} std={np.std(ss):.4f}")


if __name__ == "__main__":
    main()
