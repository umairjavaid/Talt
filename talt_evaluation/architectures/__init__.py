#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import BaseArchitecture
from .cnn import get_resnet, get_vgg, get_efficientnet
from .llm import get_bert

def get_architecture(architecture_name, dataset='cifar10', pretrained=False):
    """
    Get the specified model architecture.
    
    Args:
        architecture_name: Name of the architecture ('resnet18', 'resnet50', 'vgg16', 
                          'efficientnet-b0', 'bert-base')
        dataset: Name of the dataset this architecture will be used with
        pretrained: Whether to use pretrained weights
    
    Returns:
        tuple: (model, model_config)
    """
    architecture_name = architecture_name.lower()
    
    # CNN architectures
    if architecture_name == 'resnet18':
        return get_resnet(18, dataset, pretrained)
    elif architecture_name == 'resnet50':
        return get_resnet(50, dataset, pretrained)
    elif architecture_name == 'vgg16':
        return get_vgg(16, dataset, pretrained)
    elif architecture_name == 'efficientnet-b0':
        return get_efficientnet('b0', dataset, pretrained)
    # LLM architectures
    elif architecture_name == 'bert-base':
        return get_bert('base', dataset, pretrained)
    else:
        raise ValueError(f"Unsupported architecture: {architecture_name}")