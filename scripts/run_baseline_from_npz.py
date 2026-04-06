import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import json
import matplotlib.pyplot as plt


def save_confusion(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(
        "Loading baseline NPZ:",
        args.data_npz
        if hasattr(args, "data_npz")
        else args.__dict__.get("data_npz", args.data_npz),
    )
    d = np.load(args.data_npz, allow_pickle=True)
    X_train = d["X_train"]
    y_train = d["y_train"]
    X_test_gt = d["X_test_gt"]
    y_test_gt = d["y_test_gt"]
    X_test_recon = d["X_test_recon"]
    y_test_recon = d["y_test_recon"]
    classes = list(d["classes"])

    print(
        f"Shapes: X_train={X_train.shape} X_test_gt={X_test_gt.shape} X_test_recon={X_test_recon.shape}"
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Evaluate GT
    y_pred_gt = clf.predict(X_test_gt)
    acc_gt = float(accuracy_score(y_test_gt, y_pred_gt))
    report_gt = classification_report(
        y_test_gt, y_pred_gt, target_names=classes, output_dict=True
    )
    cm_gt = confusion_matrix(y_test_gt, y_pred_gt)

    # Evaluate recon
    y_pred_recon = clf.predict(X_test_recon)
    acc_recon = float(accuracy_score(y_test_recon, y_pred_recon))
    report_recon = classification_report(
        y_test_recon, y_pred_recon, target_names=classes, output_dict=True
    )
    cm_recon = confusion_matrix(y_test_recon, y_pred_recon)

    # Save metrics
    import csv

    with open(os.path.join(args.out, "metrics_baseline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "accuracy"])
        w.writerow(["test_gt", acc_gt])
        w.writerow(["test_recon", acc_recon])

    with open(os.path.join(args.out, "report_gt.json"), "w") as f:
        json.dump(report_gt, f, indent=2)
    with open(os.path.join(args.out, "report_recon.json"), "w") as f:
        json.dump(report_recon, f, indent=2)

    summary = {"gt": {}, "recon": {}}
    for cls in classes:
        if cls in report_gt:
            summary["gt"][cls] = {
                "f1": report_gt[cls]["f1-score"],
                "support": report_gt[cls]["support"],
            }
        if cls in report_recon:
            summary["recon"][cls] = {
                "f1": report_recon[cls]["f1-score"],
                "support": report_recon[cls]["support"],
            }
    with open(os.path.join(args.out, "report_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save confusion plots
    save_confusion(cm_gt, classes, os.path.join(args.out, "confusion_gt.png"))
    save_confusion(cm_recon, classes, os.path.join(args.out, "confusion_recon.png"))

    # Save classifier
    joblib.dump(
        {"clf": clf, "classes": classes}, os.path.join(args.out, "clf_baseline.joblib")
    )

    print(f"GT accuracy: {acc_gt:.4f}")
    print(f"Recon accuracy: {acc_recon:.4f}")


if __name__ == "__main__":
    main()
