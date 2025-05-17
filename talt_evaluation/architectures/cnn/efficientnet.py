#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models

from ..base import BaseArchitecture

class EfficientNetModel(BaseArchitecture):
    """EfficientNet model implementation with CIFAR dataset adaptations."""
    
    def __init__(self, model_variant, num_classes=10, pretrained=False):
        """
        Initialize an EfficientNet model.
        
        Args:
            model_variant: EfficientNet variant ('b0', 'b1', ..., 'b7')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(EfficientNetModel, self).__init__(f"efficientnet-{model_variant}", 'cnn')
        self.model_variant = model_variant
        self.num_classes = num_classes
        
        # Initialize the EfficientNet model
        if model_variant == 'b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
        elif model_variant == 'b1':
            base_model = models.efficientnet_b1(pretrained=pretrained)
        elif model_variant == 'b2':
            base_model = models.efficientnet_b2(pretrained=pretrained)
        elif model_variant == 'b3':
            base_model = models.efficientnet_b3(pretrained=pretrained)
        elif model_variant == 'b4':
            base_model = models.efficientnet_b4(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_variant}")
        
        # Modify the classifier for the specified number of classes
        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
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
        
        # Register hooks for key layers in the model
        # For EfficientNet, we'll monitor the output of each MBConv block
        self.model.features[1].register_forward_hook(get_activation('block1'))
        self.model.features[2].register_forward_hook(get_activation('block2'))
        self.model.features[3].register_forward_hook(get_activation('block3'))
        self.model.features[5].register_forward_hook(get_activation('block5'))
        self.model.features[7].register_forward_hook(get_activation('block7'))
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def get_optimizer_config(self):
        """Get EfficientNet-specific default TALT hyperparameters."""
        config = super().get_optimizer_config()
        
        # EfficientNet-specific adjustments
        config.update({
            'projection_dim': 80,
            'memory_size': 8,
            'update_interval': 90,
            'smoothing_factor': 0.93,
            'valley_strength': 0.08
        })
        
        return config
    
    def get_hyperparameter_search_space(self):
        """Define search space for hyperparameter tuning."""
        search_space = super().get_hyperparameter_search_space()
        
        # EfficientNet-specific search space adjustments
        search_space.update({
            'projection_dim': {'type': 'int', 'low': 24, 'high': 160},
            'memory_size': {'type': 'int', 'low': 6, 'high': 15},
            'valley_strength': {'type': 'float', 'low': 0.02, 'high': 0.3}
        })
        
        return search_space
    
    def architecture_specific_visualization(self, data):
        """Generate EfficientNet-specific visualizations."""
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


def get_efficientnet(model_variant, dataset='cifar10', pretrained=False):
    """
    Create an EfficientNet model for the specified dataset.
    
    Args:
        model_variant: EfficientNet variant ('b0', 'b1', ..., 'b7')
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
        raise ValueError(f"Unsupported dataset for EfficientNet: {dataset}")
    
    model = EfficientNetModel(model_variant, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'variant': model_variant,
        'num_classes': num_classes
    }
    
    return model, model_config