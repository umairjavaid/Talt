#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.models as models
import torch.nn as nn

class VGGModel(nn.Module):
    """VGG model implementation with CIFAR dataset adaptations."""
    
    def __init__(self, vgg_type, num_classes=10, pretrained=True):
        """
        Initialize a VGG model.
        
        Args:
            vgg_type: VGG type (e.g., 'vgg11', 'vgg11_bn', 'vgg16', 'vgg16_bn')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(VGGModel, self).__init__()
        self.name = vgg_type
        self.model_type = 'cnn'
        self.vgg_type = vgg_type
        self.num_classes = num_classes
        
        weights_value = None
        if pretrained:
            if vgg_type == 'vgg11': weights_value = models.VGG11_Weights.IMAGENET1K_V1
            elif vgg_type == 'vgg11_bn': weights_value = models.VGG11_BN_Weights.IMAGENET1K_V1
            elif vgg_type == 'vgg13': weights_value = models.VGG13_Weights.IMAGENET1K_V1
            elif vgg_type == 'vgg13_bn': weights_value = models.VGG13_BN_Weights.IMAGENET1K_V1
            elif vgg_type == 'vgg16': weights_value = models.VGG16_Weights.IMAGENET1K_V1
            elif vgg_type == 'vgg16_bn': weights_value = models.VGG16_BN_Weights.IMAGENET1K_V1
            elif vgg_type == 'vgg19': weights_value = models.VGG19_Weights.IMAGENET1K_V1
            elif vgg_type == 'vgg19_bn': weights_value = models.VGG19_BN_Weights.IMAGENET1K_V1

        if vgg_type == 'vgg11': base_model = models.vgg11(weights=weights_value)
        elif vgg_type == 'vgg11_bn': base_model = models.vgg11_bn(weights=weights_value)
        elif vgg_type == 'vgg13': base_model = models.vgg13(weights=weights_value)
        elif vgg_type == 'vgg13_bn': base_model = models.vgg13_bn(weights=weights_value)
        elif vgg_type == 'vgg16': base_model = models.vgg16(weights=weights_value)
        elif vgg_type == 'vgg16_bn': base_model = models.vgg16_bn(weights=weights_value)
        elif vgg_type == 'vgg19': base_model = models.vgg19(weights=weights_value)
        elif vgg_type == 'vgg19_bn': base_model = models.vgg19_bn(weights=weights_value)
        else:
            raise ValueError(f"Unsupported VGG type: {vgg_type}")
        
        # Modify for CIFAR: VGG models typically have a classifier for 224x224 images.
        # For 32x32 images (like CIFAR), the feature map size before classifier is different.
        # VGG output after features for 32x32 input is 512x1x1 (if avgpool is adjusted or not present before classifier)
        # PyTorch VGG models include an adaptive avg pool layer that handles this.
        
        # Replace the classifier
        num_features = base_model.classifier[0].in_features # Get in_features from the original first linear layer
        base_model.classifier = nn.Sequential(
            nn.Linear(num_features, 4096), # Standard VGG classifier dimensions
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights for the modified classifier layers
        for m in base_model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        self.model = base_model

    def forward(self, x):
        return self.model(x)

def get_vgg(vgg_type, dataset='cifar10', pretrained=False):
    """
    Create a VGG model for the specified dataset.
    
    Args:
        vgg_type: VGG type (e.g., 'vgg11', 'vgg11_bn', 'vgg16')
        dataset: Name of the dataset this model will be used with
        pretrained: Whether to use pretrained weights
    
    Returns:
        tuple: (model, model_config)
    """
    # Extract VGG variant from string like 'vgg16' -> '16'
    if vgg_type.startswith('vgg'):
        variant = vgg_type[3:]  # Remove 'vgg' prefix
        if variant.endswith('_bn'):
            vgg_variant = f"vgg{variant}"
        else:
            vgg_variant = vgg_type
    else:
        vgg_variant = vgg_type
    
    num_classes = 10
    if dataset.lower() == 'cifar100':
        num_classes = 100
    elif dataset.lower() == 'imagenet':
        num_classes = 1000
    # Add other dataset-specific num_classes if needed
    
    model = VGGModel(vgg_variant, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'vgg_type': vgg_variant, # Store the specific VGG type like vgg11_bn
        'num_classes': num_classes
    }
    
    return model, model_config
