import os
import argparse
import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, accuracy_score, classification_report
import json


def find_temperature(logits, y_true):
    def loss_fn(t):
        T = float(t[0])
        if T <= 0:
            return 1e6
        scaled = logits / T
        e = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return float(log_loss(y_true, probs))

    res = minimize(lambda x: loss_fn(x), x0=[1.0], bounds=[(0.01, 10.0)])
    return float(res.x[0])


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", required=True)
    parser.add_argument("--clf-gt", required=True)
    parser.add_argument("--clf-recon", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.data_npz, allow_pickle=True)
    X_test_gt = d["X_test_gt"]
    y_test_gt = d["y_test_gt"]
    X_test_recon = d["X_test_recon"]
    y_test_recon = d["y_test_recon"]
    classes = list(d["classes"])

    gtdata = joblib.load(args.clf_gt)
    recondata = joblib.load(args.clf_recon)

    # extract classifiers
    if isinstance(gtdata, dict) and "clfs" in gtdata:
        gt_clf = gtdata["clfs"][0]
    elif isinstance(gtdata, dict) and "clf" in gtdata:
        gt_clf = gtdata["clf"]
    else:
        gt_clf = gtdata

    if isinstance(recondata, dict) and "clf" in recondata:
        recon_clf = recondata["clf"]
    else:
        # may be saved as {'clf': clf, 'classes': classes}
        recon_clf = (
            recondata if not isinstance(recondata, dict) else recondata.get("clf")
        )

    # compute logits on test arrays
    try:
        gt_logits = gt_clf.decision_function(X_test_gt)
    except Exception:
        gt_logits = gt_clf._decision_function(X_test_gt)

    try:
        recon_logits = recon_clf.decision_function(X_test_recon)
    except Exception:
        recon_logits = recon_clf._decision_function(X_test_recon)

    # calibrate T on respective test sets
    T_gt = find_temperature(gt_logits, y_test_gt)
    T_recon = find_temperature(recon_logits, y_test_recon)
    print("Calibrated T_gt", T_gt, "T_recon", T_recon)

    scaled_gt = gt_logits / T_gt
    scaled_recon = recon_logits / T_recon

    # fuse by matching indices — assume test sets correspond by index
    min_n = min(scaled_gt.shape[0], scaled_recon.shape[0])
    fused = args.alpha * scaled_gt[:min_n] + (1.0 - args.alpha) * scaled_recon[:min_n]
    probs = softmax(fused)
    y_true = y_test_gt[:min_n]

    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs)
    rep = classification_report(y_true, preds, target_names=classes, output_dict=True)

    out = {
        "accuracy": acc,
        "log_loss": ll,
        "report": rep,
        "T_gt": T_gt,
        "T_recon": T_recon,
    }
    with open(os.path.join(args.out, "calibrated_fusion_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("Fused acc", acc, "logloss", ll)


if __name__ == "__main__":
    main()
