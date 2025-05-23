#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .cifar import get_cifar_dataset

# Wrapper functions for specific CIFAR datasets
def get_cifar10_dataset(batch_size=128, num_workers=4, root="./data", transform=None):
    """Get CIFAR-10 dataset"""
    return get_cifar_dataset(num_classes=10, root=root, transform=transform, 
                           batch_size=batch_size, num_workers=num_workers)

def get_cifar100_dataset(batch_size=128, num_workers=4, root="./data", transform=None):
    """Get CIFAR-100 dataset"""
    return get_cifar_dataset(num_classes=100, root=root, transform=transform, 
                           batch_size=batch_size, num_workers=num_workers)

# Main entry point for getting datasets
def get_dataset(dataset_name, **kwargs):
    """
    Get dataset loader based on dataset name
    
    Args:
        dataset_name: Name of the dataset (cifar10, cifar100, glue, etc.)
        **kwargs: Additional arguments to pass to the dataset loader
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if dataset_name.lower() == "cifar10":
        return get_cifar10_dataset(**kwargs)
    elif dataset_name.lower() == "cifar100":
        return get_cifar100_dataset(**kwargs)
    elif dataset_name.lower().startswith("glue"):
        try:
            from .glue import get_glue_dataset
            return get_glue_dataset(dataset_name, **kwargs)
        except ImportError:
            raise ImportError("GLUE dataset support requires additional dependencies")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

__all__ = [
    "get_dataset",
    "get_cifar10_dataset",
    "get_cifar100_dataset",
    "get_cifar_dataset",
]