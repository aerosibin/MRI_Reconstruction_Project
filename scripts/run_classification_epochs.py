import os
import argparse
import numpy as np
import cv2
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
import json
import matplotlib.pyplot as plt
import csv


def load_image_paths(root):
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


def _augment_image(im):
    # simple deterministic augment: random flip + jitter
    if np.random.rand() < 0.5:
        im = np.fliplr(im)
    if np.random.rand() < 0.5:
        im = np.flipud(im)
    if np.random.rand() < 0.3:
        j = 0.9 + 0.2 * (np.random.rand() - 0.5)
        im = np.clip(im * j, 0.0, 1.0)
    return im


def load_images(paths, size=(64, 64), augment=False):
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(p)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        im = im.astype("float32") / 255.0
        imgs.append(im.ravel())
        if augment:
            # add one augmented copy per image to increase robustness
            im2 = _augment_image(im)
            imgs.append(im2.ravel())
    return np.vstack(imgs)


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def save_plot(xs, ys, xlabel, ylabel, out_path, legends=None):
    plt.figure()
    for y, label in zip(ys, legends) if legends is not None else [(ys, None)]:
        plt.plot(xs, y, label=label)
    if legends:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test-gt", required=True)
    parser.add_argument("--test-recon", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--augment", action="store_true", help="augment training images"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train_paths, train_labels = load_image_paths(args.train)
    test_paths_gt, test_labels_gt = load_image_paths(args.test_gt)
    test_paths_recon, test_labels_recon = load_image_paths(args.test_recon)

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test_gt = le.transform(test_labels_gt)
    y_test_recon = le.transform(test_labels_recon)

    print("Loading images into memory")
    X_train = load_images(
        train_paths, size=(args.size, args.size), augment=args.augment
    )
    X_test_gt = load_images(test_paths_gt, size=(args.size, args.size))
    X_test_recon = load_images(test_paths_recon, size=(args.size, args.size))

    # compute class weights and sample weights for training
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    # map class->weight
    weight_map = {c: w for c, w in zip(classes, class_weights)}
    sample_weight = np.array([weight_map[y] for y in y_train])

    clf = SGDClassifier(
        loss="log_loss", max_iter=1, tol=None, learning_rate="optimal", random_state=42
    )

    history = {
        "epoch": [],
        "train_acc": [],
        "train_loss": [],
        "test_gt_acc": [],
        "test_gt_loss": [],
        "test_recon_acc": [],
        "test_recon_loss": [],
    }

    for epoch in range(1, args.epochs + 1):
        # partial_fit on full training set (could be minibatched)
        if epoch == 1:
            clf.partial_fit(
                X_train, y_train, classes=classes, sample_weight=sample_weight
            )
        else:
            clf.partial_fit(X_train, y_train, sample_weight=sample_weight)

        # train metrics
        train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        # obtain decision_function and softmax to approximate probabilities
        try:
            scores_train = clf.decision_function(X_train)
        except Exception:
            scores_train = clf._decision_function(X_train)
        probs_train = softmax(scores_train)
        train_loss = log_loss(y_train, probs_train, labels=classes)

        # test gt metrics
        pred_gt = clf.predict(X_test_gt)
        test_gt_acc = accuracy_score(y_test_gt, pred_gt)
        scores_gt = clf.decision_function(X_test_gt)
        probs_gt = softmax(scores_gt)
        test_gt_loss = log_loss(y_test_gt, probs_gt, labels=classes)

        # test recon metrics
        pred_recon = clf.predict(X_test_recon)
        test_recon_acc = accuracy_score(y_test_recon, pred_recon)
        scores_recon = clf.decision_function(X_test_recon)
        probs_recon = softmax(scores_recon)
        test_recon_loss = log_loss(y_test_recon, probs_recon, labels=classes)

        history["epoch"].append(epoch)
        history["train_acc"].append(float(train_acc))
        history["train_loss"].append(float(train_loss))
        history["test_gt_acc"].append(float(test_gt_acc))
        history["test_gt_loss"].append(float(test_gt_loss))
        history["test_recon_acc"].append(float(test_recon_acc))
        history["test_recon_loss"].append(float(test_recon_loss))

        print(
            f"Epoch {epoch}: train_acc={train_acc:.4f} train_loss={train_loss:.4f} test_gt_acc={test_gt_acc:.4f} test_recon_acc={test_recon_acc:.4f}"
        )

    # Save history
    with open(os.path.join(args.out, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save plots
    xs = history["epoch"]
    save_plot(
        xs,
        [history["train_acc"], history["test_gt_acc"], history["test_recon_acc"]],
        "epoch",
        "accuracy",
        os.path.join(args.out, "accuracy_curve.png"),
        legends=["train", "test_gt", "test_recon"],
    )
    save_plot(
        xs,
        [history["train_loss"], history["test_gt_loss"], history["test_recon_loss"]],
        "epoch",
        "log_loss",
        os.path.join(args.out, "loss_curve.png"),
        legends=["train", "test_gt", "test_recon"],
    )

    # Final classification reports and confusion matrices
    final_pred_gt = clf.predict(X_test_gt)
    final_report_gt = classification_report(
        y_test_gt, final_pred_gt, target_names=list(le.classes_), output_dict=True
    )
    with open(os.path.join(args.out, "report_gt_epochs.json"), "w") as f:
        json.dump(final_report_gt, f, indent=2)
    cm_gt = confusion_matrix(y_test_gt, final_pred_gt)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_gt, cmap="Blues")
    plt.colorbar()
    plt.title("confusion_gt_epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_gt_epochs.png"))
    plt.close()

    final_pred_recon = clf.predict(X_test_recon)
    final_report_recon = classification_report(
        y_test_recon, final_pred_recon, target_names=list(le.classes_), output_dict=True
    )
    with open(os.path.join(args.out, "report_recon_epochs.json"), "w") as f:
        json.dump(final_report_recon, f, indent=2)
    cm_recon = confusion_matrix(y_test_recon, final_pred_recon)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_recon, cmap="Blues")
    plt.colorbar()
    plt.title("confusion_recon_epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_recon_epochs.png"))
    plt.close()


if __name__ == "__main__":
    main()
