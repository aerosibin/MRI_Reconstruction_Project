import os
import numpy as np
import cv2
import joblib


def grad_simple(clf, x, class_idx):
    # approximate saliency for linear models: coefficient dot input
    # for ensemble average coefficients
    coefs = np.mean([c.coef_ for c in clf], axis=0)
    # x is flattened image
    sal = np.dot(coefs[class_idx], x)
    return sal.reshape((int(np.sqrt(x.size)), int(np.sqrt(x.size))))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-joblib", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = joblib.load(args.model_joblib)
    clfs = data["clfs"]
    le = data["label_encoder"]

    im = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
    x = im.astype("float32").ravel()

    # compute saliency for predicted class of first clf
    pred = clfs[0].predict(x[None, :])[0]
    sal = grad_simple(clfs, x, pred)
    # normalize
    s = sal - sal.min()
    s = (s / (s.max() + 1e-9) * 255).astype("uint8")
    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(os.path.join(args.out, "saliency.png"), s)


if __name__ == "__main__":
    main()
