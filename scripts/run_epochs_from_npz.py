import os
import argparse
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
import json
import matplotlib.pyplot as plt


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def save_plot(xs, ys, xlabel, ylabel, out_path, legends=None):
    plt.figure()
    for y, label in zip(ys, legends) if legends is not None else [(ys, None)]:
        plt.plot(xs, y, marker="o", linestyle="-", label=label)
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
    parser.add_argument("--data-npz", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.data_npz, allow_pickle=True)
    X_train = d["X_train"]
    y_train = d["y_train"]
    X_test_gt = d["X_test_gt"]
    y_test_gt = d["y_test_gt"]
    X_test_recon = d["X_test_recon"]
    y_test_recon = d["y_test_recon"]
    classes = np.unique(y_train)

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
        if epoch == 1:
            clf.partial_fit(X_train, y_train, classes=classes)
        else:
            clf.partial_fit(X_train, y_train)

        # train
        try:
            s_train = clf.decision_function(X_train)
        except Exception:
            s_train = clf._decision_function(X_train)
        probs_train = softmax(s_train)
        train_pred = probs_train.argmax(axis=1)
        train_acc = accuracy_score(y_train, train_pred)
        train_loss = log_loss(y_train, probs_train, labels=classes)

        # test gt
        try:
            s_gt = clf.decision_function(X_test_gt)
        except Exception:
            s_gt = clf._decision_function(X_test_gt)
        probs_gt = softmax(s_gt)
        pred_gt = probs_gt.argmax(axis=1)
        test_gt_acc = accuracy_score(y_test_gt, pred_gt)
        test_gt_loss = log_loss(y_test_gt, probs_gt, labels=classes)

        # test recon
        try:
            s_recon = clf.decision_function(X_test_recon)
        except Exception:
            s_recon = clf._decision_function(X_test_recon)
        probs_recon = softmax(s_recon)
        pred_recon = probs_recon.argmax(axis=1)
        test_recon_acc = accuracy_score(y_test_recon, pred_recon)
        test_recon_loss = log_loss(y_test_recon, probs_recon, labels=classes)

        history["epoch"].append(epoch)
        history["train_acc"].append(float(train_acc))
        history["train_loss"].append(float(train_loss))
        history["test_gt_acc"].append(float(test_gt_acc))
        history["test_gt_loss"].append(float(test_gt_loss))
        history["test_recon_acc"].append(float(test_recon_acc))
        history["test_recon_loss"].append(float(test_recon_loss))

        print(
            f"Epoch {epoch}: train_acc={train_acc:.4f} test_gt_acc={test_gt_acc:.4f} test_recon_acc={test_recon_acc:.4f}"
        )

    # save history and plots
    with open(os.path.join(args.out, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

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


if __name__ == "__main__":
    main()
