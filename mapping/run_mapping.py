import os
import sys
import numpy as np
import argparse

# Ensure repo root on path when executed as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mapping.signal_model import t2_signal, t1_ir_signal
from mapping.fitting import fit_t2_curve, fit_t1_ir
import cv2


def synthesize_t2_image_stack(template_img, T2_map, M0_map, TEs):
    H, W = template_img.shape
    stack = np.zeros((len(TEs), H, W), dtype=np.float32)
    for k, TE in enumerate(TEs):
        stack[k] = M0_map * np.exp(-TE / (T2_map + 1e-6))
    return stack


def synthesize_t1_ir_stack(template_img, T1_map, M0_map, TIs):
    H, W = template_img.shape
    stack = np.zeros((len(TIs), H, W), dtype=np.float32)
    for k, TI in enumerate(TIs):
        stack[k] = M0_map * (1 - 2 * np.exp(-TI / (T1_map + 1e-6)))
    return stack


def save_map_png(path, arr):
    a = np.nan_to_num(arr)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    cv2.imwrite(path, (a * 255).astype("uint8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="example image to get HxW (use a GT reconstruction PNG)",
    )
    parser.add_argument(
        "--out", type=str, default="outputs/mapping", help="output folder"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="number of worker processes for parallel T1 fitting (default: auto)",
    )
    parser.add_argument(
        "--use-bloch",
        action="store_true",
        help="use Bloch RK4 forward model for T1 fitting (more accurate, slower)",
    )
    parser.add_argument(
        "--steps-per-ms",
        type=float,
        default=1.0,
        help="integration steps per unit TI for Bloch RK4 (higher = more accurate, slower)",
    )
    parser.add_argument(
        "--inversion-eff",
        type=float,
        default=1.0,
        help="inversion efficiency used by Bloch simulation (0-1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print progress and timing information",
    )
    args = parser.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    H, W = img.shape

    os.makedirs(args.out, exist_ok=True)

    yy, xx = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    r = np.sqrt(xx**2 + yy**2)
    T2_map = 20 + 180 * (1 - r / r.max())
    M0_map = 0.8 * np.exp(-r)

    TEs = np.array([10, 30, 60, 90, 120], dtype=float)
    stack = synthesize_t2_image_stack(img, T2_map, M0_map, TEs)

    M0_fit, T2_fit = fit_t2_curve(TEs, stack)

    np.save(os.path.join(args.out, "T2_map.npy"), T2_fit)
    np.save(os.path.join(args.out, "M0_map.npy"), M0_fit)
    save_map_png(os.path.join(args.out, "T2_map.png"), T2_fit)
    save_map_png(os.path.join(args.out, "M0_map.png"), M0_fit)

    T1_map = 300 + 1000 * (1 - r / r.max())
    M0_map_t1 = 0.9 * np.exp(-r)
    TIs = np.array([50, 200, 500, 900, 1500], dtype=float)
    stack_t1 = synthesize_t1_ir_stack(img, T1_map, M0_map_t1, TIs)
    if args.verbose:
        print(
            f"Fitting T1 maps (use_bloch={args.use_bloch}) with n_workers={args.n_workers} chunk_size default"
        )
    M0_fit_t1, T1_fit = fit_t1_ir(
        TIs,
        stack_t1,
        mask=None,
        bounds=(1, 5000),
        n_workers=args.n_workers,
        use_bloch=args.use_bloch,
        steps_per_ms=args.steps_per_ms,
        inversion_eff=args.inversion_eff,
        verbose=args.verbose,
    )
    if args.verbose:
        print("Finished fitting T1 maps")

    np.save(os.path.join(args.out, "T1_map.npy"), T1_fit)
    np.save(os.path.join(args.out, "M0_map_t1.npy"), M0_fit_t1)
    save_map_png(os.path.join(args.out, "T1_map.png"), T1_fit)
    save_map_png(os.path.join(args.out, "M0_map_t1.png"), M0_fit_t1)

    print(f"Saved mapping outputs to {args.out}")


if __name__ == "__main__":
    main()
