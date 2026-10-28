"""Figures for the notebook and the report. Every figure has a title, axis labels
and is saved with a descriptive file name."""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_history(history: dict, title: str, save_path=None):
    """Training and validation loss and accuracy per epoch, best epoch marked."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="Training")
    ax_loss.plot(epochs, history["val_loss"], label="Validation")
    ax_loss.set_title("Cross-entropy loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    ax_acc.plot(epochs, history["train_acc"], label="Training")
    ax_acc.plot(epochs, history["val_acc"], label="Validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")

    for ax in (ax_loss, ax_acc):
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_model_comparison(results: dict, save_path=None, acc_ylim=(0.96, 1.0)):
    """Validation loss and validation accuracy per epoch for several models."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    for name, res in results.items():
        history = res["history"]
        epochs = range(1, len(history["val_loss"]) + 1)
        ax_loss.plot(epochs, history["val_loss"], label=name)
        ax_acc.plot(epochs, history["val_acc"], label=name)

    ax_loss.set_title("Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_ylim(0, 0.2)

    ax_acc.set_title(f"Validation accuracy (y-axis from {acc_ylim[0]})")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(*acc_ylim)
    ax_acc.legend(fontsize=8, loc="lower right")

    fig.suptitle("All models on the same validation set")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, title: str, save_path=None):
    """Confusion matrix with the count in every cell. Rows are true digits, columns predictions."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(10):
        for j in range(10):
            count = matrix[i, j]
            if count == 0:
                continue
            color = "white" if count > matrix.max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Number of images")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return matrix


def plot_misclassified(pixels, y_true, y_pred, probs, title: str, save_path=None, n_cols: int = 7):
    """Grid of every misclassified image, most confident mistakes first.

    pixels: raw (N, 784) array. Each title shows true label, predicted label and
    the model's probability for its (wrong) prediction.
    """
    wrong = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    wrong.sort(key=lambda i: probs[i, y_pred[i]], reverse=True)
    n_rows = (len(wrong) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 2.0))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(pixels[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"True {y_true[i]}, pred {y_pred[i]}\np = {probs[i, y_pred[i]]:.2f}", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return wrong
