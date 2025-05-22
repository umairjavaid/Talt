#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models

from ..base import BaseArchitecture

class VGGModel(BaseArchitecture):
    """VGG model implementation with CIFAR dataset adaptations."""
    
    def __init__(self, depth, num_classes=10, pretrained=False):
        """
        Initialize a VGG model.
        
        Args:
            depth: Depth of VGG (11, 13, 16, or 19)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(VGGModel, self).__init__(f"vgg{depth}", 'cnn')
        self.depth = depth
        self.num_classes = num_classes
        
        # Initialize the VGG model with batch normalization using updated API
        if depth == 11:
            base_model = models.vgg11_bn(weights="IMAGENET1K_V1" if pretrained else None)
        elif depth == 13:
            base_model = models.vgg13_bn(weights="IMAGENET1K_V1" if pretrained else None)
        elif depth == 16:
            base_model = models.vgg16_bn(weights="IMAGENET1K_V1" if pretrained else None)
        elif depth == 19:
            base_model = models.vgg19_bn(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unsupported VGG depth: {depth}")
        
        # Modify for CIFAR: replace fully connected layers
        base_model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),  # 1x1 feature maps for CIFAR
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        
        # Initialize weights for the modified layers
        for m in base_model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        self.model = base_model
        self.visualization_hooks = []
        self.activation_maps = {}
        
        # Register hooks for feature visualization
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations for visualization."""
        def get_activation(name):
            def hook(model, input, output):
                self.activation_maps[name] = output.detach()
            return hook
        
        # Register hooks for select layers from each block
        self.model.features[2].register_forward_hook(get_activation('block1_conv'))   # After first block's activation
        self.model.features[6].register_forward_hook(get_activation('block2_conv'))   # After second block's activation
        self.model.features[13].register_forward_hook(get_activation('block3_conv'))  # Mid third block
        self.model.features[23].register_forward_hook(get_activation('block4_conv'))  # Mid fourth block
        self.model.features[33].register_forward_hook(get_activation('block5_conv'))  # Last block
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def get_optimizer_config(self):
        """Get VGG-specific default TALT hyperparameters."""
        config = super().get_optimizer_config()
        
        # VGG has more parameters, so we adjust some hyperparameters
        config.update({
            'projection_dim': 96,
            'memory_size': 12,
            'update_interval': 80,
            'smoothing_factor': 0.92
        })
        
        return config
    
    def get_hyperparameter_search_space(self):
        """Define search space for hyperparameter tuning."""
        search_space = super().get_hyperparameter_search_space()
        
        # VGG-specific search space adjustments
        search_space.update({
            'projection_dim': {'type': 'int', 'low': 32, 'high': 192},
            'grad_store_interval': {'type': 'int', 'low': 8, 'high': 20}
        })
        
        return search_space
    
    def architecture_specific_visualization(self, data):
        """Generate VGG-specific visualizations."""
        # Forward pass to populate activation maps
        self.model.eval()
        with torch.no_grad():
            _ = self.model(data)
        
        # Prepare visualization data
        visualizations = {
            'model_type': self.model_type,
            'name': self.name,
            'activations': {},
            'feature_maps': {}
        }
        
        # Extract and process activation maps
        for layer_name, activation in self.activation_maps.items():
            # Get first image activations only
            act = activation[0].cpu()
            visualizations['activations'][layer_name] = act
            
            # Compute mean activation across channels for feature map visualization
            feature_map = act.mean(0)
            visualizations['feature_maps'][layer_name] = feature_map
        
        return visualizations


def get_vgg(depth, dataset='cifar10', pretrained=False):
    """
    Create a VGG model for the specified dataset.
    
    Args:
        depth: Depth of VGG (11, 13, 16, or 19)
        dataset: Name of the dataset this model will be used with
        pretrained: Whether to use pretrained weights
    
    Returns:
        tuple: (model, model_config)
    """
    if dataset.lower() == 'cifar10':
        num_classes = 10
    elif dataset.lower() == 'cifar100':
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset for VGG: {dataset}")
    
    model = VGGModel(depth, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'depth': depth,
        'num_classes': num_classes
    }
    
    return model, model_config