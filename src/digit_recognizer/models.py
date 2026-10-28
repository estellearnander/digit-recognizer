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


class SimpleCNN(nn.Module):
    """Plain CNN with two convolutional blocks and no regularisation.

    Block: Conv2d (3 x 3, padding 1) -> ReLU -> MaxPool2d(2).
    Shapes: [B, 1, 28, 28] -> [B, 32, 14, 14] -> [B, 64, 7, 7]
            -> flatten [B, 3136] -> [B, 128] -> [B, 10] logits.
    Kept plain on purpose so the improvements in section 4 have a clean reference.
    """

    def __init__(self, num_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
