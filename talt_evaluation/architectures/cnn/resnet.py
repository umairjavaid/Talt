#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models

from ..base import BaseArchitecture

class ResNetModel(BaseArchitecture):
    """ResNet model implementation with CIFAR dataset adaptations."""
    
    def __init__(self, depth, num_classes=10, pretrained=False):
        """
        Initialize a ResNet model.
        
        Args:
            depth: Depth of ResNet (18, 34, 50, 101, or 152)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(ResNetModel, self).__init__(f"resnet{depth}", 'cnn')
        self.depth = depth
        self.num_classes = num_classes
        
        # Initialize the ResNet model using the newer weights API
        if depth == 18:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
        elif depth == 34:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_model = models.resnet34(weights=weights)
        elif depth == 50:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
        elif depth == 101:
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            base_model = models.resnet101(weights=weights)
        elif depth == 152:
            weights = models.ResNet152_Weights.DEFAULT if pretrained else None
            base_model = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Modify the first layer to work with CIFAR's 32x32 images instead of ImageNet's 224x224
        # Replace the first 7x7 conv with kernel_size=3, stride=1, padding=1
        if num_classes <= 100:  # Assuming CIFAR dataset
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base_model.maxpool = nn.Identity()  # Remove maxpool layer
        
        # Replace the final fully connected layer
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)
        
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
        
        # Register hooks for key layers
        if self.depth <= 34:
            self.model.layer1[0].conv1.register_forward_hook(get_activation('layer1.0'))
            self.model.layer2[0].conv1.register_forward_hook(get_activation('layer2.0'))
            self.model.layer3[0].conv1.register_forward_hook(get_activation('layer3.0'))
            self.model.layer4[0].conv1.register_forward_hook(get_activation('layer4.0'))
        else:
            self.model.layer1[0].conv1.register_forward_hook(get_activation('layer1.0'))
            self.model.layer2[0].conv1.register_forward_hook(get_activation('layer2.0'))
            self.model.layer3[0].conv1.register_forward_hook(get_activation('layer3.0'))
            self.model.layer4[0].conv1.register_forward_hook(get_activation('layer4.0'))
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def get_optimizer_config(self):
        """Get ResNet-specific default TALT hyperparameters."""
        config = super().get_optimizer_config()
        
        # Adjust hyperparameters based on model depth
        if self.depth >= 50:
            config.update({
                'projection_dim': 128,
                'memory_size': 15,
                'update_interval': 120
            })
        
        return config
    
    def get_hyperparameter_search_space(self):
        """Define search space for hyperparameter tuning."""
        search_space = super().get_hyperparameter_search_space()
        
        # Adjust search space based on model depth
        if self.depth >= 50:
            search_space.update({
                'projection_dim': {'type': 'int', 'low': 32, 'high': 256},
                'memory_size': {'type': 'int', 'low': 10, 'high': 30}
            })
        
        return search_space
    
    def architecture_specific_visualization(self, data):
        """Generate ResNet-specific visualizations."""
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


def get_resnet(depth, dataset='cifar10', pretrained=False):
    """
    Create a ResNet model for the specified dataset.
    
    Args:
        depth: Depth of ResNet (18, 34, 50, 101, or 152)
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
        raise ValueError(f"Unsupported dataset for ResNet: {dataset}")
    
    model = ResNetModel(depth, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'depth': depth,
        'num_classes': num_classes
    }
    
    return model, model_config