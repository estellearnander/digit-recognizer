"""Neural network models for 28 x 28 grayscale digit classification.

Every model takes images of shape [B, 1, 28, 28] and returns logits of
shape [B, 10]. No softmax is applied: nn.CrossEntropyLoss expects logits.
"""

import torch
from torch import nn

N_PIXELS = 784
N_CLASSES = 10


class MLP(nn.Module):
    """Multilayer perceptron baseline (lecture 5 pattern).

    Flatten -> Linear -> ReLU -> Dropout for each hidden layer -> Linear to 10 logits.
    The Flatten is inside the model so the MLP uses the same [B, 1, 28, 28]
    batches as the CNNs.
    """

    def __init__(self, hidden_dims=(256, 128), dropout: float = 0.2, num_classes: int = N_CLASSES):
        super().__init__()
        layers = [nn.Flatten()]
        prev_dim = N_PIXELS
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
