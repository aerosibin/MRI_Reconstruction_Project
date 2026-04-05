import os
import shutil


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def copy_suffix(src_root, dst_root, suffix_from, suffix_to=None):
    ensure_dir(dst_root)
    for cls in sorted(os.listdir(src_root)):
        p = os.path.join(src_root, cls)
        if not os.path.isdir(p):
            continue
        out_cls = os.path.join(dst_root, cls)
        ensure_dir(out_cls)
        for f in sorted(os.listdir(p)):
            if not f.lower().endswith(".png"):
                continue
            if f.endswith(suffix_from):
                src = os.path.join(p, f)
                dst_name = f if suffix_to is None else f.replace(suffix_from, suffix_to)
                dst = os.path.join(out_cls, dst_name)
                shutil.copyfile(src, dst)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs/inference_test")
    parser.add_argument("--baseline-root", default="outputs/inference_baseline_demo")
    parser.add_argument("--out-train", default="data/tmp_train")
    parser.add_argument("--out-test-gt", default="data/tmp_test_gt")
    parser.add_argument("--out-test-recon", default="data/tmp_test_recon")
    args = parser.parse_args()

    # Build train from baseline demo GT images (files ending with _gt.png)
    copy_suffix(args.baseline_root, args.out_train, "_gt.png", "_gt.png")

    # Build test gt and recon from inference_test
    copy_suffix(args.outputs_root, args.out_test_gt, "_gt.png", "_gt.png")
    copy_suffix(args.outputs_root, args.out_test_recon, "_pred.png", "_pred.png")

    print("Prepared datasets:")
    print(" train ->", args.out_train)
    print(" test_gt ->", args.out_test_gt)
    print(" test_recon ->", args.out_test_recon)
