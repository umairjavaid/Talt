#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import BaseArchitecture
import torchvision.models as models
import torch.nn as nn

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
        
        weights_value = None
        if pretrained:
            if depth == 18: weights_value = models.ResNet18_Weights.IMAGENET1K_V1
            elif depth == 34: weights_value = models.ResNet34_Weights.IMAGENET1K_V1
            elif depth == 50: weights_value = models.ResNet50_Weights.IMAGENET1K_V1
            elif depth == 101: weights_value = models.ResNet101_Weights.IMAGENET1K_V1
            elif depth == 152: weights_value = models.ResNet152_Weights.IMAGENET1K_V1
        
        if depth == 18:
            base_model = models.resnet18(weights=weights_value)
        elif depth == 34:
            base_model = models.resnet34(weights=weights_value)
        elif depth == 50:
            base_model = models.resnet50(weights=weights_value)
        elif depth == 101:
            base_model = models.resnet101(weights=weights_value)
        elif depth == 152:
            base_model = models.resnet152(weights=weights_value)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Modify the first layer for CIFAR-like images (32x32)
        # Original ResNet conv1 is nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # For CIFAR, a smaller kernel and stride is common.
        if num_classes <= 100: # Assuming CIFAR10/100 or similar small images
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base_model.maxpool = nn.Identity() # Remove max pooling after conv1 for small images
        
        # Replace the final fully connected layer
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)
        
        self.model = base_model
        # self.visualization_hooks = [] # Example for hooks, if needed
        # self.activation_maps = {}   # Example for hooks, if needed
        # self._register_hooks()      # Call if hooks are implemented
    
    # def _register_hooks(self):
    #     pass # Implement if needed
    
    def forward(self, x):
        return self.model(x)

    # get_optimizer_config and get_hyperparameter_search_space can be inherited or overridden
    # architecture_specific_visualization can be inherited or overridden

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
    num_classes = 10
    if dataset.lower() == 'cifar100':
        num_classes = 100
    elif dataset.lower() == 'imagenet': # Example for a different dataset
        num_classes = 1000
    # Add other dataset-specific num_classes if needed
    
    model = ResNetModel(depth, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'depth': depth,
        'num_classes': num_classes
    }
    
    return model, model_config
