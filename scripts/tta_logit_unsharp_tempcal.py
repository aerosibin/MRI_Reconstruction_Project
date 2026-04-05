import os
import numpy as np
import cv2
import joblib
from scipy.ndimage import laplace


def tta_transforms(im):
    outs = [im]
    outs.append(np.fliplr(im))
    outs.append(np.flipud(im))
    for a in (-5, 5):
        M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), a, 1.0)
        outs.append(
            cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)
        )
    return outs


def unsharp_laplacian(im, amount=1.0):
    # im expected 0..1 float
    lap = laplace(im)
    enhanced = im + amount * lap
    enhanced = np.clip(enhanced, 0.0, 1.0)
    return enhanced


def predict_with_logit_averaging(model, image_paths, size=(64, 64), amount=1.0, T=1.0):
    # model: a wrapper with decision_function
    all_probs = []
    for p in image_paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        im = im.astype("float32") / 255.0
        # apply unsharp laplacian to refocus
        im = unsharp_laplacian(im, amount=amount)
        tts = tta_transforms(im)
        logits_list = []
        for t in tts:
            x = t.ravel()[None, :]
            logits = model.decision_function(x)
            logits_list.append(logits)
        # stack logits: shape (n_transforms, n_classes)
        stacked = np.vstack(logits_list)
        # take mean of logits (logit averaging)
        mean_logits = np.mean(stacked, axis=0, keepdims=True)
        # temperature scaling
        scaled = mean_logits / float(T)
        e = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        all_probs.append(probs.ravel())
    return np.vstack(all_probs)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-joblib", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    # load model - support joblib or torch pth via meta
    if args.model_joblib.endswith(".pth"):
        meta = joblib.load(args.model_joblib.replace(".pth", "_meta.joblib"))
        classes = meta["classes"]
        from scripts.model_wrapper import TorchWrapper

        model = TorchWrapper(args.model_joblib, classes)
    else:
        data = joblib.load(args.model_joblib)
        model = data["clfs"][0]

    paths = []
    for root, dirs, files in os.walk(args.images):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))

    # compute logits and probs; save both so downstream tools can use raw logits if desired
    all_logits = []
    all_probs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (args.size, args.size), interpolation=cv2.INTER_AREA)
        im = im.astype("float32") / 255.0
        im = unsharp_laplacian(im, amount=args.amount)
        tts = tta_transforms(im)
        logits_list = []
        for t in tts:
            x = t.ravel()[None, :]
            logits = model.decision_function(x)
            logits_list.append(logits)
        stacked = np.vstack(logits_list)
        mean_logits = np.mean(stacked, axis=0, keepdims=True)
        scaled = mean_logits / float(args.T)
        e = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        all_logits.append(scaled.ravel())
        all_probs.append(probs.ravel())

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(
        {
            "paths": paths,
            "probs": np.vstack(all_probs),
            "logits": np.vstack(all_logits),
        },
        os.path.join(args.out, "tta_logit_probs.joblib"),
    )
    joblib.dump(
        {"paths": paths, "logits": np.vstack(all_logits)},
        os.path.join(args.out, "tta_logit_logits.joblib"),
    )


if __name__ == "__main__":
    main()
