import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Ensure repo root is on python path so `models` and `utils` import correctly
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
from utils.image_postproc import unsharp_mask
from utils.fft import fft2c, ifft2c


def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    # assume state_dict
    if isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
        except Exception:
            # maybe ckpt contains extra keys (e.g. 'model_state_dict')
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                # try filtering
                sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
                model.load_state_dict(sd)
    else:
        # ckpt is a full model object
        model = ckpt
    return model


def to_np(x):
    return x.detach().cpu().numpy()


def _gather_sequence_paths(src):
    # If src contains subdirectories with frames, return list of folders.
    # Otherwise return [src] so MRIDataset can be used on it.
    entries = [os.path.join(src, p) for p in os.listdir(src)]
    # treat as sequence folder if there are directories inside
    dirs = [p for p in entries if os.path.isdir(p)]
    if dirs:
        # return directories that contain image files
        sequences = []
        for d in dirs:
            # check if directory contains image files
            imgs = [
                f
                for f in os.listdir(d)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
            ]
            if imgs:
                sequences.append(d)
        if sequences:
            return sequences
    # otherwise, treat src itself as a single dataset (may contain class subfolders)
    return [src]


def run_on_folder(folder, model, device, out_dir, limit=None):
    # Use existing MRIDataset to build inputs from folder structure
    ds = MRIDataset(folder)
    pin_memory = True if device.type == "cuda" else False
    loader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=pin_memory)

    records = []

    for idx, (x, y) in enumerate(loader):
        if limit is not None and idx >= limit:
            break

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            # Test-time augmentation: original + hflip + vflip
            preds = []
            # optionally use autocast if on CUDA for faster inference
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = model(x)
            else:
                out = model(x)
            preds.append(out)
            # horizontal flip
            x_h = torch.flip(x, dims=[3])
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out_h = model(x_h)
            else:
                out_h = model(x_h)
            out_h = torch.flip(out_h, dims=[3])
            preds.append(out_h)
            # vertical flip
            x_v = torch.flip(x, dims=[2])
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out_v = model(x_v)
            else:
                out_v = model(x_v)
            out_v = torch.flip(out_v, dims=[2])
            preds.append(out_v)
            # average
            pred = torch.mean(torch.stack(preds, dim=0), dim=0)

        # Convert tensors to numpy images in [0,1]
        pred_np = to_np(pred[:, 0, :, :])[0]
        gt_np = to_np(y[:, 0, :, :])[0]

        inp = to_np(x)[0]  # shape (2,H,W)
        zf = np.sqrt(inp[0] ** 2 + inp[1] ** 2)

        # Clip and scale
        pred_img = np.clip(pred_np, 0.0, 1.0)
        gt_img = np.clip(gt_np, 0.0, 1.0)
        zf_img = np.clip(zf, 0.0, 1.0)

        # Compute metrics
        try:
            p = psnr_sk(gt_img, pred_img, data_range=1.0)
        except Exception:
            p = float("nan")
        try:
            s = ssim_sk(gt_img, pred_img, data_range=1.0)
        except Exception:
            s = float("nan")

        records.append({"idx": idx, "psnr": float(p), "ssim": float(s)})

        # Save images as PNG (0-255)
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
        # Data-consistency step (one iteration) in k-space to restore measured lines
        # Build complex images from zero-filled input and model prediction
        inp = to_np(x)[0]  # (2,H,W)
        zf_complex = inp[0] + 1j * inp[1]
        k_meas = fft2c(zf_complex)
        mask = np.abs(k_meas) > 0

        pred_complex = pred_img.astype(np.complex64)  # assume zero phase
        k_pred = fft2c(pred_complex)
        # combine measured k-space and predicted
        k_comb = np.where(mask, k_meas, k_pred)
        dc_img = np.abs(ifft2c(k_comb))
        dc_img = np.clip(dc_img.real, 0.0, 1.0)

        # sharpen the data-consistent image for presentation
        pred_sharp = unsharp_mask(dc_img, kernel_size=3, sigma=0.8, strength=0.7)
        cv2.imwrite(
            os.path.join(out_dir, f"{idx:04d}_pred_sharp.png"),
            (pred_sharp * 255).astype("uint8"),
        )
        # also save the DC-corrected prediction
        cv2.imwrite(
            os.path.join(out_dir, f"{idx:04d}_pred_dc.png"),
            (dc_img * 255).astype("uint8"),
        )

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Run UNet inference on processed images (supports sequence folders)"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/processed/train",
        help="source folder containing images or sequences",
    )
    parser.add_argument(
        "--out", type=str, default="outputs/inference", help="output folder"
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="limit images per sequence (default 50)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to run on, e.g. cpu or cuda"
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="use CUDA AMP for inference if running on cuda",
    )
    parser.add_argument(
        "--ckpt", type=str, default="model.pth", help="model checkpoint path"
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Model
    model = UNet().to(device)

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    model = load_checkpoint(ckpt_path, model, device)
    model.eval()

    sequences = _gather_sequence_paths(args.src)

    # If top-level has multiple sequences, create per-sequence output subfolders
    aggregate_records = []
    for seq in sequences:
        seq_name = os.path.basename(seq.rstrip(os.sep))
        seq_out = os.path.join(args.out, seq_name)
        os.makedirs(seq_out, exist_ok=True)

        print(f"Processing sequence/folder: {seq} -> {seq_out}")
        records = run_on_folder(seq, model, device, seq_out, limit=args.limit)
        # save per-sequence metrics
        import csv

        metrics_path = os.path.join(seq_out, "metrics.csv")
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "psnr", "ssim"])
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        aggregate_records.extend([{"sequence": seq_name, **r} for r in records])

    # Save aggregate metrics
    if aggregate_records:
        import csv

        metrics_path = os.path.join(args.out, "metrics_aggregate.csv")
        keys = [k for k in aggregate_records[0].keys()]
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in aggregate_records:
                writer.writerow(r)

        # Print high level summary
        ps = [r["psnr"] for r in aggregate_records if not np.isnan(r["psnr"])]
        ss = [r["ssim"] for r in aggregate_records if not np.isnan(r["ssim"])]
        if ps:
            print(
                f"Processed {len(aggregate_records)} images. PSNR mean={np.mean(ps):.3f} std={np.std(ps):.3f}"
            )
        if ss:
            print(f"SSIM mean={np.mean(ss):.4f} std={np.std(ss):.4f}")


if __name__ == "__main__":
    main()
