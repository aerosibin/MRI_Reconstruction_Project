import os
import json
import matplotlib.pyplot as plt


def plot_with_markers(xs, ys_list, labels, out_path, xlabel, ylabel):
    plt.figure()
    for y, label in zip(ys_list, labels):
        plt.plot(xs, y, marker="o", linestyle="-", label=label)
    if labels:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    root = os.path.join("outputs", "analysis")
    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, "training_history.json")
    if not os.path.exists(json_path):
        print(f"training_history.json not found at {json_path}")
        return
    with open(json_path, "r") as f:
        history = json.load(f)

    xs = history.get("epoch", [])
    if not xs:
        print("No epochs found in history")
        return

    accs = [
        history.get("train_acc", []),
        history.get("test_gt_acc", []),
        history.get("test_recon_acc", []),
    ]
    losses = [
        history.get("train_loss", []),
        history.get("test_gt_loss", []),
        history.get("test_recon_loss", []),
    ]

    plot_with_markers(
        xs,
        accs,
        ["train", "test_gt", "test_recon"],
        os.path.join(root, "accuracy_curve.png"),
        "epoch",
        "accuracy",
    )
    plot_with_markers(
        xs,
        losses,
        ["train", "test_gt", "test_recon"],
        os.path.join(root, "loss_curve.png"),
        "epoch",
        "log_loss",
    )


if __name__ == "__main__":
    main()
