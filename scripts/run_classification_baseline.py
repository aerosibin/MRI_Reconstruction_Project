import os
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import csv


def load_image_paths(root):
    # expects class subfolders
    paths = []
    labels = []
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                paths.append(os.path.join(p, f))
                labels.append(cls)
    return paths, labels


def load_images(paths, size=(64, 64)):
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(p)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        im = im.astype("float32") / 255.0
        imgs.append(im.ravel())
    return np.vstack(imgs)


def save_confusion(cm, labels, out_path):
    import matplotlib.pyplot as plt

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
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test-gt", type=str, required=True)
    parser.add_argument("--test-recon", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/analysis")
    parser.add_argument("--size", type=int, default=64, help="downsample size (square)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading train paths")
    train_paths, train_labels = load_image_paths(args.train)
    print(f"Found {len(train_paths)} train images")
    print("Loading test (GT) paths")
    test_paths_gt, test_labels_gt = load_image_paths(args.test_gt)
    print(f"Found {len(test_paths_gt)} test GT images")
    print("Loading test (recon) paths")
    test_paths_recon, test_labels_recon = load_image_paths(args.test_recon)
    print(f"Found {len(test_paths_recon)} test recon images")

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test_gt = le.transform(test_labels_gt)
    y_test_recon = le.transform(test_labels_recon)

    print("Loading images (this may take a moment)")
    X_train = load_images(train_paths, size=(args.size, args.size))
    X_test_gt = load_images(test_paths_gt, size=(args.size, args.size))
    X_test_recon = load_images(test_paths_recon, size=(args.size, args.size))

    print("Training logistic regression baseline")
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="saga")
    clf.fit(X_train, y_train)

    # Evaluate on GT
    y_pred_gt = clf.predict(X_test_gt)
    acc_gt = accuracy_score(y_test_gt, y_pred_gt)
    report_gt = classification_report(
        y_test_gt, y_pred_gt, target_names=le.classes_, output_dict=True
    )
    cm_gt = confusion_matrix(y_test_gt, y_pred_gt)

    # Evaluate on recon
    y_pred_recon = clf.predict(X_test_recon)
    acc_recon = accuracy_score(y_test_recon, y_pred_recon)
    report_recon = classification_report(
        y_test_recon, y_pred_recon, target_names=le.classes_, output_dict=True
    )
    cm_recon = confusion_matrix(y_test_recon, y_pred_recon)

    # Save metrics
    with open(os.path.join(args.out, "metrics_baseline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "accuracy"])
        w.writerow(["test_gt", acc_gt])
        w.writerow(["test_recon", acc_recon])

    # Save classification reports
    import json

    with open(os.path.join(args.out, "report_gt.json"), "w") as f:
        json.dump(report_gt, f, indent=2)
    with open(os.path.join(args.out, "report_recon.json"), "w") as f:
        json.dump(report_recon, f, indent=2)

    # Save confusion matrices
    save_confusion(cm_gt, le.classes_, os.path.join(args.out, "confusion_gt.png"))
    save_confusion(cm_recon, le.classes_, os.path.join(args.out, "confusion_recon.png"))

    print(f"GT accuracy: {acc_gt:.4f}")
    print(f"Recon accuracy: {acc_recon:.4f}")


if __name__ == "__main__":
    main()
