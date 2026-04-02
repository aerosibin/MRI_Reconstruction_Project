import os
import sys
import argparse
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as psnr_sk,
    structural_similarity as ssim_sk,
)

# ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.visualization import make_montage
import csv


def load_image(path):
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype("float32") / 255.0


def evaluate_folder(pred_folder, out_folder, top_k=5):
    os.makedirs(out_folder, exist_ok=True)

    # Find pred images and corresponding gt/zf
    preds = sorted([p for p in os.listdir(pred_folder) if p.endswith("_pred.png")])

    records = []
    for p in preds:
        idx = p.replace("_pred.png", "")
        pred_path = os.path.join(pred_folder, p)
        gt_path = os.path.join(pred_folder, f"{idx}_gt.png")
        zf_path = os.path.join(pred_folder, f"{idx}_zf.png")

        try:
            pred = load_image(pred_path)
            gt = load_image(gt_path)
        except FileNotFoundError:
            continue

        ps = float(psnr_sk(gt, pred, data_range=1.0))
        ss = float(ssim_sk(gt, pred, data_range=1.0))

        records.append(
            {
                "idx": idx,
                "psnr": ps,
                "ssim": ss,
                "pred": pred_path,
                "gt": gt_path,
                "zf": zf_path,
            }
        )

    # Save CSV
    csv_path = os.path.join(out_folder, "evaluation.csv")
    keys = ["idx", "psnr", "ssim"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in keys})

    # Summary
    ps = [r["psnr"] for r in records]
    ss = [r["ssim"] for r in records]
    summary_path = os.path.join(out_folder, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"N={len(records)}\n")
        if ps:
            f.write(f"PSNR mean={np.mean(ps):.3f} std={np.std(ps):.3f}\n")
        if ss:
            f.write(f"SSIM mean={np.mean(ss):.4f} std={np.std(ss):.4f}\n")

    # Create montages for best and worst (by psnr)
    if records:
        records_sorted = sorted(records, key=lambda r: r["psnr"])  # low->high
        worst = records_sorted[:top_k]
        best = records_sorted[-top_k:]

        worst_montage = make_montage(
            [r["gt"] for r in worst],
            [r["zf"] for r in worst],
            [r["pred"] for r in worst],
        )
        best_montage = make_montage(
            [r["gt"] for r in best], [r["zf"] for r in best], [r["pred"] for r in best]
        )

        import cv2

        cv2.imwrite(
            os.path.join(out_folder, "montage_worst.png"),
            (worst_montage * 255).astype("uint8"),
        )
        cv2.imwrite(
            os.path.join(out_folder, "montage_best.png"),
            (best_montage * 255).astype("uint8"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against GT (PSNR/SSIM)"
    )
    parser.add_argument(
        "--pred-folder",
        type=str,
        required=True,
        help="folder containing *_pred.png and *_gt.png",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/evaluation",
        help="output folder for evaluation results",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="number of best/worst images to montage"
    )
    args = parser.parse_args()

    evaluate_folder(args.pred_folder, args.out, top_k=args.top_k)


if __name__ == "__main__":
    main()
