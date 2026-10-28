"""Figures for the notebook and the report. Every figure has a title, axis labels
and is saved with a descriptive file name."""

import matplotlib.pyplot as plt


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
