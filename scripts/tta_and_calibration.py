import os
import numpy as np
import cv2
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def tta_transforms(im):
    # return list of images: original, hflip, vflip, rotate +-5 degrees
    outs = [im]
    outs.append(np.fliplr(im))
    outs.append(np.flipud(im))
    for a in (-5, 5):
        M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), a, 1.0)
        outs.append(
            cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)
        )
    return outs


def load_images(paths, size=(64, 64)):
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        im = im.astype("float32") / 255.0
        imgs.append(im.ravel())
    return np.vstack(imgs)


def predict_with_tta(joblib_path, image_paths, size=(64, 64)):
    data = joblib.load(joblib_path)
    clfs = data["clfs"]
    le = data["label_encoder"]

    X = []
    for p in image_paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        tts = tta_transforms(im)
        # obtain ensemble average across transforms and classifiers
        probs_accum = None
        for t in tts:
            x = t.ravel()[None, :]
            probs_t = []
            for clf in clfs:
                s = clf.decision_function(x)
                e = np.exp(s - np.max(s, axis=1, keepdims=True))
                probs_t.append(e / e.sum(axis=1, keepdims=True))
            probs_t = np.mean(probs_t, axis=0)
            if probs_accum is None:
                probs_accum = probs_t
            else:
                probs_accum += probs_t
        probs_accum /= len(tts)
        X.append(probs_accum.ravel())
    return np.vstack(X)


def calibrate_probs(probs, labels):
    # Platt scaling (logistic regression) on top of probs
    # Train/val split for calibration
    X_train, X_val, y_train, y_val = train_test_split(
        probs, labels, test_size=0.2, random_state=42
    )
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    probs_cal = lr.predict_proba(X_val)
    print("Calibration holdout log_loss:", log_loss(y_val, probs_cal))
    return lr


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-joblib", required=True)
    parser.add_argument(
        "--images", required=True, help="directory of class subfolders or CSV list"
    )
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()

    # load image paths (flat directory of images assumed for simplicity)
    paths = []
    for root, dirs, files in os.walk(args.images):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))

    probs = predict_with_tta(args.model_joblib, paths)
    # naive label parse from folder name (last folder before filename)
    labels = [os.path.basename(os.path.dirname(p)) for p in paths]
    # map to integers using label encoder saved in joblib (not available here), so skip detailed calibration
    # Save raw probs for offline calibration
    os.makedirs(args.out, exist_ok=True)
    joblib.dump(
        {"paths": paths, "probs": probs}, os.path.join(args.out, "tta_probs.joblib")
    )


if __name__ == "__main__":
    main()
