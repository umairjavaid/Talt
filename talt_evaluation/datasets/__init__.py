#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .cifar import get_cifar_dataset
from .glue import get_glue_dataset

def get_dataset(dataset_name, batch_size=128, num_workers=4):
    """
    Get train, validation, and test data loaders for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'glue-sst2')
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar_dataset(10, batch_size, num_workers)
    elif dataset_name.lower() == 'cifar100':
        return get_cifar_dataset(100, batch_size, num_workers)
    elif dataset_name.lower() == 'glue-sst2':
        return get_glue_dataset('sst2', batch_size, num_workers)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")