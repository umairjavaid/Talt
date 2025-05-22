#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base import BaseArchitecture
from talt.model import SimpleCNN

def get_architecture(architecture_name, dataset_name="cifar10", pretrained=False, **kwargs):
    """
    Get architecture model by name
    
    Args:
        architecture_name (str): Name of the architecture
        dataset_name (str): Name of the dataset
        pretrained (bool): Whether to use pretrained weights
        **kwargs: Additional arguments for the model
        
    Returns:
        model: The architecture model
    """
    # Determine number of classes based on dataset
    num_classes = 10  # default
    if dataset_name.lower() == "cifar100":
        num_classes = 100
    
    # Copy kwargs to avoid modifying the original
    model_kwargs = kwargs.copy()
    
    # Prevent dataset from being passed to model constructors
    if 'dataset' in model_kwargs:
        del model_kwargs['dataset']
    
    if architecture_name.lower() == "simplecnn":
        return SimpleCNN(num_classes=num_classes, **model_kwargs)
    elif architecture_name.lower() == "resnet18":
        # Use weights parameter instead of pretrained
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights, **model_kwargs)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif architecture_name.lower() == "mobilenetv2":
        # Use weights parameter instead of pretrained
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights, **model_kwargs)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

__all__ = ["get_architecture", "BaseArchitecture"]