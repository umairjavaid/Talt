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
        self.name = 'simplecnn'
        self.model_type = 'cnn'
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_classes = num_classes
        
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
    
    def get_optimizer_config(self):
        """Get optimizer-specific configurations for this architecture."""
        return {
            'cnn': {
                'lr': 0.01,
                'weight_decay': 5e-4,
                'momentum': 0.9
            },
            'talt': {
                'lr': 0.01,
                'projection_dim': 16,
                'memory_size': 8,
                'update_interval': 25,
                'valley_strength': 0.15,
                'smoothing_factor': 0.3,
                'grad_store_interval': 5,
                'cov_decay': 0.95,
                'adaptive_reg': True
            }
        }
    
    def get_hyperparameter_search_space(self):
        """Get hyperparameter search space for this architecture."""
        return {
            'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-1, 'log': True},
            'projection_dim': {'type': 'int', 'low': 8, 'high': 32},
            'memory_size': {'type': 'int', 'low': 5, 'high': 15},
            'update_interval': {'type': 'int', 'low': 10, 'high': 50},
            'valley_strength': {'type': 'float', 'low': 0.05, 'high': 0.5},
            'smoothing_factor': {'type': 'float', 'low': 0.1, 'high': 0.7},
            'grad_store_interval': {'type': 'int', 'low': 3, 'high': 10},
            'cov_decay': {'type': 'float', 'low': 0.9, 'high': 0.99}
        }
    
    def architecture_specific_visualization(self, inputs):
        """Generate architecture-specific visualizations."""
        self.eval()
        feature_maps = {}
        
        with torch.no_grad():
            # Get feature maps from the convolutional layers
            x = inputs
            for i, layer in enumerate(self.features):
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    feature_maps[f'conv_{i}'] = x[0]  # First sample in batch
        
        return {
            'feature_maps': feature_maps,
            'input_images': inputs
        }
