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
            -> flatten [B, 3136] -> [B, head_dim] -> [B, 10] logits.
    Kept plain on purpose so the improvements in section 4 have a clean reference.

    head_dim sets the size of the hidden dense layer (128 by default). With
    head_dim=64 the CNN has fewer parameters than the MLP (size-matched control).
    """

    def __init__(self, head_dim: int = 128, num_classes: int = N_CLASSES):
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
            nn.Linear(64 * 7 * 7, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class SimpleCNNGAP(nn.Module):
    """Control model: SimpleCNN's two conv blocks with a global-average-pooling head.

    Each of the 64 feature maps is averaged to one number, so the head is a
    single Linear(64 -> 10). About 19,000 parameters in total, fewer than the MLP.
    Used to test whether the CNN beats the MLP only because it has more parameters.
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
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ImprovedCNN(nn.Module):
    """SimpleCNN plus BatchNorm and dropout. Same convolution widths and head size.

    Block: Conv2d (3 x 3, padding 1, no bias) -> BatchNorm2d -> ReLU -> MaxPool2d(2) -> Dropout.
    The convolution has no bias because BatchNorm's learned shift does the same job.
    Head: Flatten -> Linear(3136 -> 128) -> ReLU -> Dropout -> Linear(128 -> 10).
    """

    def __init__(self, conv_dropout: float = 0.25, head_dropout: float = 0.5, num_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=conv_dropout),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=conv_dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
