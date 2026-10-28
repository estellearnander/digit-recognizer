"""Data checks, split and Dataset output shapes, on small synthetic data."""

import numpy as np
import pandas as pd
import pytest
import torch

from digit_recognizer import DigitDataset, check_data, compute_mean_std, make_augmentation, split_train_val


def make_frames():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.integers(0, 256, size=(42000, 784), dtype=np.uint8),
                         columns=[f"pixel{i}" for i in range(784)])
    train.insert(0, "label", np.arange(42000) % 10)
    test = pd.DataFrame(rng.integers(0, 256, size=(28000, 784), dtype=np.uint8),
                        columns=[f"pixel{i}" for i in range(784)])
    return train, test


def test_check_data_accepts_valid_data():
    train, test = make_frames()
    assert check_data(train, test) is True


def test_check_data_rejects_label_outside_0_to_9():
    train, test = make_frames()
    train.loc[0, "label"] = 10
    with pytest.raises(ValueError):
        check_data(train, test)


def test_split_is_stratified_90_10():
    train, _ = make_frames()
    X_train, X_val, y_train, y_val = split_train_val(train, val_size=0.1, seed=42)
    assert len(X_train) == 37800 and len(X_val) == 4200
    assert (np.bincount(y_val) == 420).all()


def test_dataset_item_has_shape_1_28_28_and_int64_label():
    pixels = np.random.default_rng(0).integers(0, 256, size=(8, 784))
    labels = np.arange(8) % 10
    mean, std = compute_mean_std(pixels)
    ds = DigitDataset(pixels, labels, mean, std)
    image, label = ds[0]
    assert image.shape == (1, 28, 28)
    assert image.dtype == torch.float32
    assert label.dtype == torch.int64


def test_test_dataset_returns_only_images():
    pixels = np.zeros((3, 784))
    ds = DigitDataset(pixels, None, 0.0, 1.0)
    assert ds[0].shape == (1, 28, 28)


def test_augmentation_keeps_the_image_shape():
    image = torch.rand(1, 28, 28)
    assert make_augmentation()(image).shape == (1, 28, 28)
