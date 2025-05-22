#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base import BaseArchitecture
from talt.model import SimpleCNN
from .cnn.resnet import get_resnet
from .cnn.vgg import get_vgg
from .cnn.efficientnet import get_efficientnet

def get_architecture(architecture_name, dataset="cifar10", **kwargs):
    """
    Get architecture model and configuration by name
    
    Args:
        architecture_name (str): Name of the architecture
        dataset (str): Name of the dataset
        **kwargs: Additional arguments for the model
        
    Returns:
        tuple: (model, model_config)
    """
    # Parse ResNet depth from architecture name
    if architecture_name.lower().startswith("resnet"):
        try:
            depth = int(architecture_name.lower().replace("resnet", ""))
            return get_resnet(depth, dataset=dataset, **kwargs)
        except ValueError:
            raise ValueError(f"Invalid ResNet architecture: {architecture_name}")
    
    # Handle VGG models
    elif architecture_name.lower().startswith("vgg"):
        return get_vgg(architecture_name.lower(), dataset=dataset, **kwargs)
    
    # Handle EfficientNet models
    elif architecture_name.lower().startswith("efficientnet"):
        model_variant = architecture_name.lower()
        return get_efficientnet(model_variant, dataset=dataset, **kwargs)
    
    # Handle SimpleCNN
    elif architecture_name.lower() == "simplecnn":
        model = SimpleCNN(num_classes=10 if dataset.lower() == "cifar10" else 100, **kwargs)
        model_config = {
            'name': 'simplecnn',
            'model_type': 'cnn',
            'num_classes': 10 if dataset.lower() == "cifar10" else 100
        }
        return model, model_config
    
    # Handle any other architecture
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

__all__ = ["get_architecture", "BaseArchitecture"]