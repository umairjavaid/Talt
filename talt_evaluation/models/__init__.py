#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CNN models
from .cnn import SimpleCNN # Assuming cnn.py contains SimpleCNN
from .resnet import ResNetModel, get_resnet
from .vgg import VGGModel, get_vgg
from .efficientnet import EfficientNetModel, get_efficientnet

# LLM models
from .bert import BERTModel, get_bert

import torch

def get_architecture(architecture_name, dataset="cifar10", **kwargs):
    """
    Get architecture model and configuration by name.
    This function now acts as a dispatcher to the specific get_* functions
    for each model type, which are located in their respective files.
    
    Args:
        architecture_name (str): Name of the architecture (e.g., "resnet18", "vgg11_bn", "bert-base")
        dataset (str): Name of the dataset
        **kwargs: Additional arguments for the model
        
    Returns:
        tuple: (model, model_config)
    """
    arch_lower = architecture_name.lower()

    if arch_lower.startswith("resnet"):
        try:
            # Assumes architecture_name is like "resnet18", "resnet50"
            depth = int(arch_lower.replace("resnet", ""))
            return get_resnet(depth, dataset=dataset, **kwargs)
        except ValueError:
            raise ValueError(f"Invalid ResNet architecture: {architecture_name}. Expected format like resnet18.")
    
    elif arch_lower.startswith("vgg"):
        # Assumes architecture_name is like "vgg11", "vgg16_bn"
        return get_vgg(arch_lower, dataset=dataset, **kwargs)
    
    elif arch_lower.startswith("efficientnet"):
        # Assumes architecture_name is like "efficientnet-b0"
        model_variant = arch_lower.split('-')[-1] # e.g., 'b0' from 'efficientnet-b0'
        return get_efficientnet(model_variant, dataset=dataset, **kwargs)
    
    elif arch_lower.startswith("bert"):
        # Assumes architecture_name is like "bert-base", "bert-large"
        variant = arch_lower.split('-')[-1] # e.g., 'base' from 'bert-base'
        return get_bert(variant, dataset=dataset, **kwargs)

    elif arch_lower == "simplecnn":
        # Determine num_channels and image_size based on dataset
        if dataset.lower() in ["cifar10", "cifar100"]:
            num_channels = 3
            image_size = 32
            num_classes = 10 if dataset.lower() == "cifar10" else 100
        elif dataset.lower() == "mnist":
            num_channels = 1
            image_size = 28
            num_classes = 10
        else:
            # Default values if dataset not recognized
            num_channels = kwargs.pop('num_channels', 3)
            image_size = kwargs.pop('image_size', 32)
            num_classes = kwargs.pop('num_classes', 10)
        
        # Allow kwargs to override defaults for SimpleCNN if provided
        num_channels = kwargs.pop('num_channels', num_channels)
        image_size = kwargs.pop('image_size', image_size)
        num_classes = kwargs.pop('num_classes', num_classes)

        model = SimpleCNN(num_channels=num_channels, image_size=image_size, num_classes=num_classes)
        
        # Add required attributes for compatibility with the framework
        model.name = 'simplecnn'
        model.model_type = 'cnn'
        
        model_config = {
            'name': 'simplecnn',
            'model_type': 'cnn',
            'num_classes': num_classes,
            'num_channels': num_channels,
            'image_size': image_size
        }
        return model, model_config
    
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

def get_dataset(dataset_name, **kwargs):
    """
    Get dataset loader by name.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "cifar10", "mnist").
        **kwargs: Additional arguments for the dataset loader.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Import from the datasets module
    from talt_evaluation.datasets import get_dataset as get_ds
    return get_ds(dataset_name, **kwargs)

__all__ = ["get_architecture", 
           "SimpleCNN", "ResNetModel", "get_resnet", 
           "VGGModel", "get_vgg", "EfficientNetModel", "get_efficientnet",
           "BERTModel", "get_bert",
           "get_dataset"]
