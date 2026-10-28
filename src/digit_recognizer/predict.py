"""Predictions from trained models."""

import numpy as np
import pandas as pd
import torch

from .models import ImprovedCNN


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


def write_submission(predictions, save_path) -> pd.DataFrame:
    """Write the Kaggle submission file: header ImageId,Label, ImageId from 1."""
    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions) + 1),
        "Label": np.asarray(predictions, dtype=int),
    })
    submission.to_csv(save_path, index=False)
    return submission


def validate_submission(path, n_rows: int = 28000) -> bool:
    """Check the submission file against the assignment's requirements. Raise ValueError if not met."""
    submission = pd.read_csv(path)
    if list(submission.columns) != ["ImageId", "Label"]:
        raise ValueError(f"Header must be ImageId,Label, got {list(submission.columns)}")
    if len(submission) != n_rows:
        raise ValueError(f"Expected {n_rows} rows, got {len(submission)}")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if not (submission["ImageId"].values == np.arange(1, n_rows + 1)).all():
        raise ValueError(f"ImageId must run from 1 to {n_rows} in order")
    if not pd.api.types.is_integer_dtype(submission["Label"]):
        raise ValueError("Labels must be integers")
    if submission["Label"].min() < 0 or submission["Label"].max() > 9:
        raise ValueError("Labels must be between 0 and 9")
    return True


def ensemble_from_weights(weight_paths, test_loader, device) -> np.ndarray:
    """Average the class probabilities of saved ImprovedCNN models (model D).

    Rebuilds each model, loads its saved weights and predicts the loader.
    Used to reproduce the submitted predictions without retraining.
    """
    all_probs = []
    for path in weight_paths:
        model = ImprovedCNN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        all_probs.append(predict_proba(model, test_loader, device))
    return np.mean(all_probs, axis=0)
