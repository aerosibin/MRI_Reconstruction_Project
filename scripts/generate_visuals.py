import os
import sys
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def make_row(img_paths, thumb=256):
    imgs = []
    for p in img_paths:
        if not os.path.exists(p):
            imgs.append(np.zeros((thumb, thumb), dtype=np.uint8))
            continue
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (thumb, thumb), interpolation=cv2.INTER_AREA)
        imgs.append(im)
    return np.hstack(imgs)


def save_montage_for_sequence(seq_dir, out_dir, thumb=256):
    # gather indices by scanning *_pred.png files
    preds = sorted([p for p in os.listdir(seq_dir) if p.endswith("_pred.png")])
    if not preds:
        return
    os.makedirs(out_dir, exist_ok=True)
    for idx_file in preds:
        idx = idx_file.replace("_pred.png", "")
        gt = os.path.join(seq_dir, f"{idx}_gt.png")
        zf = os.path.join(seq_dir, f"{idx}_zf.png")
        pred = os.path.join(seq_dir, f"{idx}_pred.png")
        pred_sharp = os.path.join(seq_dir, f"{idx}_pred_sharp.png")

        row = make_row([gt, zf, pred, pred_sharp], thumb=thumb)
        outp = os.path.join(out_dir, f"{idx}_montage.png")
        cv2.imwrite(outp, row)


def save_colormap_map(np_path, out_path, cmap="viridis"):
    import matplotlib.pyplot as plt

    arr = np.load(np_path)
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(arr, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_vis = os.path.join(root, "outputs", "visuals")
    os.makedirs(out_vis, exist_ok=True)

    # sequences
    seq_dirs = [
        d
        for d in glob(os.path.join(root, "outputs", "inference*", "*"))
        if os.path.isdir(d) and any(f.endswith("_pred.png") for f in os.listdir(d))
    ]
    for sd in seq_dirs:
        name = os.path.basename(sd.rstrip(os.sep))
        seq_out = os.path.join(out_vis, name)
        os.makedirs(seq_out, exist_ok=True)
        save_montage_for_sequence(sd, seq_out, thumb=256)

    # mapping maps (np and png)
    map_dir = os.path.join(root, "outputs", "mapping_demo")
    if os.path.isdir(map_dir):
        for f in os.listdir(map_dir):
            if f.endswith(".npy"):
                arrp = os.path.join(map_dir, f)
                outp = os.path.join(out_vis, f.replace(".npy", ".png"))
                save_colormap_map(arrp, outp, cmap="inferno")
    print("Saved visuals to", out_vis)


if __name__ == "__main__":
    main()
