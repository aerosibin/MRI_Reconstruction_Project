import os
import argparse
import numpy as np
import cv2
import torch

# ensure repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, REPO_ROOT)

from models.unet import UNet
from simulation.cine_simulation import synthesize_moving_gaussian
from simulation.undersample import cartesian_undersample_kspace


def save_movie(frames, outpath, fps=10):
    h, w = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(outpath, fourcc, fps, (w, h), isColor=False)
    for f in frames:
        vw.write((f * 255).astype("uint8"))
    vw.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/cine_demo")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--accel", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ckpt", default="model.pth")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    os.makedirs(args.out, exist_ok=True)

    # synthesize stack
    stack = synthesize_moving_gaussian(H=args.size, W=args.size, T=args.frames)

    # undersample
    k_under, mask = cartesian_undersample_kspace(stack, accel=args.accel)

    # zero-filled recon
    zf_stack = np.zeros_like(stack)
    for t in range(args.frames):
        k = k_under[t]
        im = np.fft.ifft2(np.fft.ifftshift(k))
        zf_stack[t] = np.abs(im)

    # load model
    model = UNet().to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict):
            try:
                model.load_state_dict(ckpt)
            except Exception:
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
        else:
            model = ckpt
    model.eval()

    pred_stack = np.zeros_like(stack)
    with torch.no_grad():
        for t in range(args.frames):
            inp = zf_stack[t]
            # create two-channel input (real, imag) using single-channel magnitude (no phase)
            x = np.stack([inp, np.zeros_like(inp)], axis=0)[None]
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            out = model(x_t)
            pred = out.detach().cpu().numpy()[0, 0]
            pred_stack[t] = np.clip(pred, 0.0, 1.0)

    # save a few example frames and movie
    # write as PNG frames and an MP4
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for t in range(args.frames):
        cv2.imwrite(
            os.path.join(frames_dir, f"{t:03d}_gt.png"),
            (stack[t] * 255).astype("uint8"),
        )
        cv2.imwrite(
            os.path.join(frames_dir, f"{t:03d}_zf.png"),
            (zf_stack[t] * 255).astype("uint8"),
        )
        cv2.imwrite(
            os.path.join(frames_dir, f"{t:03d}_pred.png"),
            (pred_stack[t] * 255).astype("uint8"),
        )

    save_movie(stack, os.path.join(args.out, "gt.mp4"))
    save_movie(zf_stack, os.path.join(args.out, "zf.mp4"))
    save_movie(pred_stack, os.path.join(args.out, "pred.mp4"))

    print("Saved cine demo to", args.out)


if __name__ == "__main__":
    main()
