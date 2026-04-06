import os
import argparse
import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, accuracy_score, classification_report
import json


def find_temperature(logits, y_true):
    # logits: (N, C) raw logits (not softmaxed)
    # y_true: (N,) integer labels

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-logits", required=True)
    parser.add_argument("--recon-logits", required=True)
    parser.add_argument("--gt-labels", required=True)
    parser.add_argument("--recon-labels", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    gt = joblib.load(args.gt_logits)
    recon = joblib.load(args.recon_logits)

    # Expect dict with 'logits' and 'paths'
    gt_logits = np.vstack(gt.get("logits"))
    recon_logits = np.vstack(recon.get("logits"))
    gt_paths = gt.get("paths")
    recon_paths = recon.get("paths")

    # labels files are joblibs with integer arrays or dicts
    y_gt = joblib.load(args.gt_labels)
    y_recon = joblib.load(args.recon_labels)

    # calibrate Ts
    print("Calibrating GT logits...")
    T_gt = find_temperature(gt_logits, np.array(y_gt))
    print("T_gt", T_gt)
    print("Calibrating RECON logits...")
    T_recon = find_temperature(recon_logits, np.array(y_recon))
    print("T_recon", T_recon)

    # apply temperatures and fuse by weighted logit avg
    scaled_gt = gt_logits / T_gt
    scaled_recon = recon_logits / T_recon

    # align lengths: assume same ordering; if lengths differ, find common paths
    # Build mapping path->index for recon
    recon_map = {p: i for i, p in enumerate(recon_paths)}
    fused_probs = []
    fused_logits = []
    y_fused = []
    for i, p in enumerate(gt_paths):
        if p not in recon_map:
            continue
        j = recon_map[p]
        lg = scaled_gt[i]
        lr = scaled_recon[j]
        combined = args.alpha * lg + (1.0 - args.alpha) * lr
        e = np.exp(combined - np.max(combined))
        probs = e / e.sum()
        fused_probs.append(probs)
        fused_logits.append(combined)
        y_fused.append(y_gt[i])

    fused_probs = np.vstack(fused_probs)
    y_fused = np.array(y_fused)

    # evaluate
    preds = fused_probs.argmax(axis=1)
    acc = accuracy_score(y_fused, preds)
    ll = log_loss(y_fused, fused_probs)
    report = classification_report(y_fused, preds, output_dict=True)
    out = {
        "accuracy": acc,
        "log_loss": ll,
        "report": report,
        "T_gt": T_gt,
        "T_recon": T_recon,
    }
    with open(os.path.join(args.out, "calibrated_fusion_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Fused accuracy", acc, "log_loss", ll)


if __name__ == "__main__":
    main()
