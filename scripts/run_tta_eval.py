import os
import joblib
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import json


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


def predict_probs_for_dir(clfs, paths, size=(64, 64)):
    all_probs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        tts = tta_transforms(im)
        probs_acc = None
        for t in tts:
            x = t.ravel()[None, :]
            probs_list = []
            for clf in clfs:
                try:
                    s = clf.decision_function(x)
                except Exception:
                    s = clf._decision_function(x)
                e = np.exp(s - np.max(s, axis=1, keepdims=True))
                probs_list.append(e / e.sum(axis=1, keepdims=True))
            probs_t = np.mean(probs_list, axis=0)
            if probs_acc is None:
                probs_acc = probs_t
            else:
                probs_acc += probs_t
        probs_acc /= len(tts)
        all_probs.append(probs_acc.ravel())
    return np.vstack(all_probs)


def gather_paths_labels(root):
    paths = []
    labels = []
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            paths.append(os.path.join(p, f))
            labels.append(cls)
    return paths, labels


def plot_confusion(cm, classes, outpath, title="confusion"):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-joblib", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test-gt", required=True)
    parser.add_argument("--test-recon", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data = joblib.load(args.model_joblib)
    clfs = data["clfs"]
    le = data["label_encoder"]
    classes = list(le.classes_)

    print("Gathering train paths for calibration...")
    train_paths, train_labels = gather_paths_labels(args.train)
    # compute probs on train for calibration
    print("Predicting train probs with TTA (this may take a while)...")
    probs_train = predict_probs_for_dir(clfs, train_paths, size=(args.size, args.size))
    y_train = le.transform(train_labels)

    # fit Platt scaling logistic regression on train probs
    print("Fitting Platt scaling logistic regression...")
    lr = LogisticRegression(max_iter=2000)
    lr.fit(probs_train, y_train)

    # Evaluate on test_gt
    print("Evaluating on test_gt...")
    test_gt_paths, test_gt_labels = gather_paths_labels(args.test_gt)
    probs_gt = predict_probs_for_dir(clfs, test_gt_paths, size=(args.size, args.size))
    probs_gt_cal = lr.predict_proba(probs_gt)
    y_gt = le.transform(test_gt_labels)
    ypred_gt = probs_gt_cal.argmax(axis=1)
    acc_gt = accuracy_score(y_gt, ypred_gt)
    ll_gt = log_loss(y_gt, probs_gt_cal, labels=range(len(classes)))
    cm_gt = confusion_matrix(y_gt, ypred_gt)
    report_gt = classification_report(
        y_gt, ypred_gt, target_names=classes, output_dict=True
    )

    # Evaluate on test_recon
    print("Evaluating on test_recon...")
    test_recon_paths, test_recon_labels = gather_paths_labels(args.test_recon)
    probs_recon = predict_probs_for_dir(
        clfs, test_recon_paths, size=(args.size, args.size)
    )
    probs_recon_cal = lr.predict_proba(probs_recon)
    y_recon = le.transform(test_recon_labels)
    ypred_recon = probs_recon_cal.argmax(axis=1)
    acc_recon = accuracy_score(y_recon, ypred_recon)
    ll_recon = log_loss(y_recon, probs_recon_cal, labels=range(len(classes)))
    cm_recon = confusion_matrix(y_recon, ypred_recon)
    report_recon = classification_report(
        y_recon, ypred_recon, target_names=classes, output_dict=True
    )

    out = {
        "accuracy_gt": float(acc_gt),
        "logloss_gt": float(ll_gt),
        "accuracy_recon": float(acc_recon),
        "logloss_recon": float(ll_recon),
        "report_gt": report_gt,
        "report_recon": report_recon,
    }

    with open(os.path.join(args.out, "tta_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    plot_confusion(
        cm_gt,
        classes,
        os.path.join(args.out, "confusion_gt_tta.png"),
        title="confusion_gt_tta",
    )
    plot_confusion(
        cm_recon,
        classes,
        os.path.join(args.out, "confusion_recon_tta.png"),
        title="confusion_recon_tta",
    )

    print("Done. Metrics written to", args.out)


if __name__ == "__main__":
    main()
