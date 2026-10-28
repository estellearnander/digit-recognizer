"""Loading, checking, splitting and serving the Kaggle Digit Recognizer data."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

N_PIXELS = 784
IMAGE_SIZE = 28


def load_train(data_path: str | Path) -> pd.DataFrame:
    """Load train.csv (label + 784 pixel columns)."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. See data/README.md.")
    return pd.read_csv(data_path)


def load_test(data_path: str | Path) -> pd.DataFrame:
    """Load test.csv (784 pixel columns, no label)."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. See data/README.md.")
    return pd.read_csv(data_path)


def check_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Check that both files have the expected structure. Raise ValueError if not."""
    if train_df.shape != (42000, N_PIXELS + 1):
        raise ValueError(f"train.csv should be 42000 x 785, got {train_df.shape}")
    if test_df.shape != (28000, N_PIXELS):
        raise ValueError(f"test.csv should be 28000 x 784, got {test_df.shape}")
    if train_df.columns[0] != "label":
        raise ValueError(f"First column of train.csv should be 'label', got {train_df.columns[0]!r}")
    if "label" in test_df.columns:
        raise ValueError("test.csv should not contain a label column")
    if train_df.isna().any().any() or test_df.isna().any().any():
        raise ValueError("The data contains missing values")

    labels = train_df["label"]
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError(f"Labels should be 0 to 9, got {labels.min()} to {labels.max()}")

    for name, pixels in [("train", train_df.iloc[:, 1:]), ("test", test_df)]:
        if pixels.values.min() < 0 or pixels.values.max() > 255:
            raise ValueError(f"{name} pixels should be between 0 and 255")
    return True


def split_train_val(train_df: pd.DataFrame, val_size: float = 0.1, seed: int = 42):
    """Stratified hold-out split.

    Returns X_train, X_val, y_train, y_val as NumPy arrays.
    X arrays have shape (n, 784) with raw pixel values 0-255.
    """
    X = train_df.drop(columns="label").values
    y = train_df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=seed
    )
    return X_train, X_val, y_train, y_val


def compute_mean_std(pixels: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of the pixels after scaling to [0, 1].

    Call this on the training pixels only, so validation and test data
    never influence the preprocessing.
    """
    scaled = pixels.astype(np.float32) / 255.0
    return float(scaled.mean()), float(scaled.std())


class DigitDataset(Dataset):
    """PyTorch Dataset of 28 x 28 grayscale digits.

    Each item is (image, label) with image shape [1, 28, 28] (float32)
    and label int64, or only the image when no labels are given (test set).

    Order of operations in __getitem__:
      1. scale to [0, 1]
      2. augmentation (only if a transform is given)
      3. normalise with the training mean and std
    Augmentation runs before normalisation so that the empty area it
    creates is filled with 0, which means black background only before
    normalising.
    """

    def __init__(self, pixels: np.ndarray, labels: np.ndarray | None = None,
                 mean: float = 0.0, std: float = 1.0, transform=None):
        images = torch.tensor(pixels, dtype=torch.float32) / 255.0
        self.images = images.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.int64)
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        image = (image - self.mean) / self.std
        if self.labels is None:
            return image
        return image, self.labels[idx]


def make_loaders(train_ds: Dataset, val_ds: Dataset, batch_size: int = 128, seed: int = 42):
    """DataLoaders for training (shuffled, seeded) and validation (fixed order).

    num_workers=0 because multiprocess loading inside Jupyter on macOS can hang,
    and the data already sits in memory.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def make_augmentation(degrees: float = 10, shift: float = 0.1, scale=(0.9, 1.1)):
    """Random affine augmentation for training images (tensors of shape [1, 28, 28] in [0, 1]).

    Small rotations, shifts and zooms only. No flips: a mirrored digit is no longer
    the same digit, and large rotations could turn a 6 into a 9.
    Empty areas are filled with 0, which is black background before normalisation.
    """
    return transforms.RandomAffine(
        degrees=degrees,
        translate=(shift, shift),
        scale=scale,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=0,
    )
