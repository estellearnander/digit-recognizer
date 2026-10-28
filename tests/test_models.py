"""Every model takes images of shape [B, 1, 28, 28] and returns logits of shape [B, 10]."""

import pytest
import torch

from digit_recognizer import MLP, ImprovedCNN, SimpleCNN, SimpleCNNGAP, count_parameters

MODELS = {
    "MLP": lambda: MLP(),
    "SimpleCNN": lambda: SimpleCNN(),
    "SimpleCNN-64": lambda: SimpleCNN(head_dim=64),
    "SimpleCNNGAP": lambda: SimpleCNNGAP(),
    "ImprovedCNN": lambda: ImprovedCNN(),
}


@pytest.mark.parametrize("name", MODELS)
def test_model_returns_logits_of_shape_batch_by_10(name):
    model = MODELS[name]()
    model.eval()
    images = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (4, 10)


@pytest.mark.parametrize("name, expected", [
    ("MLP", 235_146),
    ("SimpleCNN", 421_642),
    ("SimpleCNN-64", 220_234),
    ("SimpleCNNGAP", 19_466),
    ("ImprovedCNN", 421_738),
])
def test_parameter_counts_match_the_report(name, expected):
    assert count_parameters(MODELS[name]()) == expected
