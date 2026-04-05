import os
import cv2
import numpy as np


def edge_enhance_dir(src_dir, out_dir, method="laplacian", size=(256, 256)):
    os.makedirs(out_dir, exist_ok=True)
    for cls in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, cls)
        if not os.path.isdir(p):
            continue
        out_cls = os.path.join(out_dir, cls)
        os.makedirs(out_cls, exist_ok=True)
        for f in sorted(os.listdir(p)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            im = cv2.imread(os.path.join(p, f), cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
            if method == "laplacian":
                edges = cv2.Laplacian(im, cv2.CV_64F)
                edges = np.abs(edges)
            else:
                # sobel
                sx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sx * sx + sy * sy)
            # normalize and merge with original to enhance
            edges = (edges - edges.min()) / (edges.ptp() + 1e-9)
            imf = im.astype("float32") / 255.0
            enhanced = np.clip(imf + edges, 0.0, 1.0)
            out_path = os.path.join(out_cls, f)
            cv2.imwrite(out_path, (enhanced * 255).astype("uint8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["laplacian", "sobel"], default="laplacian")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    edge_enhance_dir(
        args.src, args.out, method=args.method, size=(args.size, args.size)
    )
