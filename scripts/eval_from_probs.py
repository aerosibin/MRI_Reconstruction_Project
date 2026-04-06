import os
import joblib
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt


def gather_labels_from_paths(paths, le):
    labels = [os.path.basename(os.path.dirname(p)) for p in paths]
    if le is None:
        # build label encoder from observed labels
        classes = sorted(list(set(labels)))
        mapc = {c: i for i, c in enumerate(classes)}
        y = np.array([mapc[l] for l in labels])
        return y, classes
    else:
        y = le.transform(labels)
        return np.array(y), list(le.classes_)


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
    parser.add_argument("--probs-joblib", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()

    data = joblib.load(args.probs_joblib)
    paths = data["paths"]
    probs = data["probs"]
    # load label encoder if exists
    le_path = os.path.join(args.out, "clfs_ensemble.joblib")
    le = None
    if os.path.exists(le_path):
        try:
            tmp = joblib.load(le_path)
            le = tmp.get("label_encoder", None)
        except Exception:
            le = None

    y, classes = gather_labels_from_paths(paths, le)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs, labels=range(len(classes)))
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, target_names=classes, output_dict=True)

    out = {"accuracy": float(acc), "log_loss": float(ll), "report": report}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tta_logit_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    plot_confusion(
        cm,
        classes,
        os.path.join(args.out, "confusion_tta_logit.png"),
        title="confusion_tta_logit",
    )
    print("Written metrics and confusion to", args.out)


if __name__ == "__main__":
    main()
