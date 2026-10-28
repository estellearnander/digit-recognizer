"""Predictions from trained models."""

import numpy as np
import torch


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    """Class probabilities for every image in the loader, shape (N, 10).

    The loader may yield (images, labels) or only images (test set).
    Softmax is applied here, at prediction time only: the training loss works on logits.
    """
    model.to(device)
    model.eval()
    all_probs = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)
