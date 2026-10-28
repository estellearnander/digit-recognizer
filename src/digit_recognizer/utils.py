"""Small helpers shared by the data, model and training code."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix the random seeds of Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Return the best available device: CUDA GPU, Apple GPU (MPS) or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
