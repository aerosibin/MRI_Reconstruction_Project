import os
import argparse
import json
import shutil
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def im_stats(im):
    return {
        "min": float(np.min(im)),
        "max": float(np.max(im)),
        "mean": float(np.mean(im)),
        "std": float(np.std(im)),
        "p10": float(np.percentile(im, 10)),
        "p50": float(np.percentile(im, 50)),
        "p90": float(np.percentile(im, 90)),
    }


def read_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    im = im.astype("float32") / 255.0
    return im


def save_hist(data, outpath, title):
    plt.figure()
    plt.hist(data, bins=256, range=(0, 1), color="tab:blue", alpha=0.8)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def collect_from_folder(root, num_samples=100):
    imgs = []
    paths = []
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if not os.path.isdir(p):
            continue
        files = glob(os.path.join(p, "*.png"))
        files = sorted(files)[:num_samples]
        for f in files:
            imgs.append(read_gray(f))
            paths.append(f)
    return imgs, paths


def collect_recon(root, num_samples=100):
    # structured: outputs/inference_test/<class>/*_{gt,zf,pred}.png
    data = {"gt": [], "zf": [], "pred": [], "paths": {"gt": [], "zf": [], "pred": []}}
    if not os.path.exists(root):
        return data
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if not os.path.isdir(p):
            continue
        files = sorted(glob(os.path.join(p, "*_gt.png")))[:num_samples]
        for f in files:
            base = os.path.splitext(f)[0]
            gt = f
            zf = base + "_zf.png" if os.path.exists(base + "_zf.png") else None
            pred = base + "_pred.png" if os.path.exists(base + "_pred.png") else None
            if gt and os.path.exists(gt):
                data["gt"].append(read_gray(gt))
                data["paths"]["gt"].append(gt)
            if zf and os.path.exists(zf):
                data["zf"].append(read_gray(zf))
                data["paths"]["zf"].append(zf)
            if pred and os.path.exists(pred):
                data["pred"].append(read_gray(pred))
                data["paths"]["pred"].append(pred)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train")
    parser.add_argument("--test", default="data/processed/test")
    parser.add_argument("--recon", default="outputs/inference_test")
    parser.add_argument("--out", default="outputs/debug")
    parser.add_argument("--num", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    summary = {"train": {}, "test": {}, "recon": {}}

    # Train
    if os.path.exists(args.train):
        imgs, paths = collect_from_folder(args.train, num_samples=args.num)
        allpix = np.concatenate([im.ravel() for im in imgs]) if imgs else np.array([])
        summary["train"]["count"] = len(imgs)
        if imgs:
            summary["train"]["stats"] = im_stats(allpix)
            save_hist(
                allpix,
                os.path.join(args.out, "train_hist.png"),
                "Train pixel histogram",
            )
            # save a few examples
            exdir = os.path.join(args.out, "train_examples")
            os.makedirs(exdir, exist_ok=True)
            for i, p in enumerate(paths[:10]):
                shutil.copy(p, os.path.join(exdir, os.path.basename(p)))
    else:
        summary["train"]["count"] = 0

    # Test
    if os.path.exists(args.test):
        imgs, paths = collect_from_folder(args.test, num_samples=args.num)
        allpix = np.concatenate([im.ravel() for im in imgs]) if imgs else np.array([])
        summary["test"]["count"] = len(imgs)
        if imgs:
            summary["test"]["stats"] = im_stats(allpix)
            save_hist(
                allpix, os.path.join(args.out, "test_hist.png"), "Test pixel histogram"
            )
            exdir = os.path.join(args.out, "test_examples")
            os.makedirs(exdir, exist_ok=True)
            for i, p in enumerate(paths[:10]):
                shutil.copy(p, os.path.join(exdir, os.path.basename(p)))
    else:
        summary["test"]["count"] = 0

    # Reconstructions
    recon = collect_recon(args.recon, num_samples=args.num)
    for key in ["gt", "zf", "pred"]:
        imgs = recon.get(key, [])
        summary["recon"][key] = {"count": len(imgs)}
        if imgs:
            allpix = np.concatenate([im.ravel() for im in imgs])
            summary["recon"][key]["stats"] = im_stats(allpix)
            save_hist(
                allpix,
                os.path.join(args.out, f"recon_{key}_hist.png"),
                f"Recon {key} pixel histogram",
            )
            exdir = os.path.join(args.out, f"recon_{key}_examples")
            os.makedirs(exdir, exist_ok=True)
            for i, p in enumerate(recon["paths"][key][:10]):
                shutil.copy(p, os.path.join(exdir, os.path.basename(p)))

    # Save summary
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote debug outputs to", args.out)


if __name__ == "__main__":
    main()
