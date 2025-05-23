#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import BaseArchitecture
import torchvision.models as models
import torch.nn as nn

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
        
        weights_value = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        
        if model_variant == 'b0':
            base_model = models.efficientnet_b0(weights=weights_value)
        elif model_variant == 'b1':
            base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_variant == 'b2':
            base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_variant == 'b3':
            base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_variant == 'b4':
            base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        # Add other variants if needed, e.g., b5, b6, b7
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_variant}")
        
        # Modify the classifier for the specified number of classes
        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        self.model = base_model

    def forward(self, x):
        return self.model(x)
    
    # get_optimizer_config and get_hyperparameter_search_space can be inherited or overridden
    # architecture_specific_visualization can be inherited or overridden

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
    num_classes = 10
    if dataset.lower() == 'cifar100':
        num_classes = 100
    elif dataset.lower() == 'imagenet': # Example for a different dataset
        num_classes = 1000
    # Add other dataset-specific num_classes if needed
    
    model = EfficientNetModel(model_variant, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'variant': model_variant,
        'num_classes': num_classes
    }
    
    return model, model_config
