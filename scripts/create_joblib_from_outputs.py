import os
import joblib
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def gather_images(root, size=(64, 64)):
    paths = []
    labels = []
    X = []
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fp = os.path.join(p, f)
            im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
            im = im.astype("float32") / 255.0
            X.append(im.ravel())
            paths.append(fp)
            labels.append(cls)
    return np.vstack(X) if X else np.zeros((0, size[0] * size[1])), labels


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="outputs/inference_test")
    parser.add_argument("--out", default="outputs/analysis/clfs_ensemble.joblib")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    X, labels = gather_images(args.images, size=(args.size, args.size))
    if X.shape[0] == 0:
        raise SystemExit("No images found in " + args.images)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # train a logistic regression on flattened images
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    joblib.dump({"clfs": [clf], "label_encoder": le}, args.out)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
