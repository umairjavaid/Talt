#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base import BaseArchitecture
from talt.model import SimpleCNN

def get_architecture(architecture_name, dataset_name="cifar10", **kwargs):
    """
    Get architecture model by name
    
    Args:
        architecture_name (str): Name of the architecture
        dataset_name (str): Name of the dataset
        **kwargs: Additional arguments for the model
        
    Returns:
        model: The architecture model
    """
    num_classes = 10  # default
    if dataset_name.lower() == "cifar100":
        num_classes = 100
    
    if architecture_name.lower() == "simplecnn":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif architecture_name.lower() == "resnet18":
        model = models.resnet18(pretrained=False, **kwargs)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif architecture_name.lower() == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=False, **kwargs)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

__all__ = ["get_architecture", "BaseArchitecture"]