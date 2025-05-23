#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Add current directory to path for relative imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from .cifar import get_cifar10, get_cifar100
from .mnist_loader import get_mnist

def get_dataset(dataset_name, batch_size=128, num_workers=4, root="./data"):
    """
    Get dataset loader by name.
    
    Args:
        dataset_name (str): Name of the dataset
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        root (str): Root directory for the dataset
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataset_lower = dataset_name.lower()
    
    if dataset_lower == "cifar10":
        return get_cifar10(batch_size=batch_size, num_workers=num_workers, root=root)
    elif dataset_lower == "cifar100":
        return get_cifar100(batch_size=batch_size, num_workers=num_workers, root=root)
    elif dataset_lower == "mnist":
        return get_mnist(batch_size=batch_size, num_workers=num_workers, data_dir=root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

__all__ = ['get_dataset', 'get_cifar10', 'get_cifar100', 'get_mnist']