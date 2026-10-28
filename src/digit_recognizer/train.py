"""Training and evaluation loops.

The order of operations follows the Session 9 notebook:
zero_grad -> forward -> loss on logits -> backward -> optimizer step.
"""

import copy
import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one pass over the loader. Returns (average loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate without updating weights. Returns (average loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, epochs: int, device, lr: float = 1e-3, scheduler: str | None = None):
    """Train with Adam and cross-entropy, validating after every epoch.

    At the end the model is restored to the weights of the epoch with the
    best validation accuracy. Returns a history dict.

    scheduler: None for a constant learning rate, or "cosine" to decrease it
    from lr towards 0 over the epochs (CosineAnnealingLR, as in Session 12).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = None
    if scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler is not None:
        raise ValueError(f"Unknown scheduler: {scheduler!r}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_epoch = 0
    best_weights = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        history["lr"].append(optimizer.param_groups[0]["lr"])
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if lr_scheduler is not None:
            lr_scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:2d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f}")

    model.load_state_dict(best_weights)
    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_val_acc
    history["train_time_s"] = time.time() - start
    print(f"Best validation accuracy {best_val_acc:.4f} at epoch {best_epoch} "
          f"({history['train_time_s']:.0f} s)")
    return history
