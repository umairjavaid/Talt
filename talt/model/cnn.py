"""Neural network model for TALT."""

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for image classification.

    Features a convolutional feature extractor followed by a fully-connected
    classifier with dropout for regularization.
    """
    def __init__(self, num_channels: int, image_size: int, num_classes: int = 10):
        super().__init__()
        # Convolutional layers with BatchNorm
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Calculate flattened dimension after convolutions
        flat_dim = (image_size // 4) ** 2 * 64

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)
