#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Provide higher level dataset access functions
from .cifar import get_cifar10_dataset, get_cifar100_dataset
# Import the following modules only when needed to avoid circular imports
# from .glue import get_glue_dataset

def get_dataset(name, **kwargs):
    """
    Main entry point for getting datasets.
    
    Args:
        name: Dataset name (cifar10, cifar100, glue-sst2, etc.)
        **kwargs: Additional arguments for dataset loading
    
    Returns:
        train_loader, val_loader, test_loader: Data loaders for train, validation, and test sets
    """
    if name == 'cifar10':
        return get_cifar10_dataset(**kwargs)
    elif name == 'cifar100':
        return get_cifar100_dataset(**kwargs)
    elif name.startswith('glue-'):
        # Import only when needed to avoid circular import
        from .glue import get_glue_dataset
        return get_glue_dataset(name[5:], **kwargs)  # Remove 'glue-' prefix
    else:
        raise ValueError(f"Dataset {name} not supported")