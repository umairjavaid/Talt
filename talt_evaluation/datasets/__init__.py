#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Add current directory to path for relative imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .cifar import get_cifar10, get_cifar100
    from .glue import get_glue_sst2
except ImportError:
    # Fallback for direct execution
    try:
        from cifar import get_cifar10, get_cifar100
        from glue import get_glue_sst2
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import dataset modules: {e}")
        # Provide dummy functions
        def get_cifar10(*args, **kwargs):
            raise NotImplementedError("CIFAR10 dataset not available")
        def get_cifar100(*args, **kwargs):
            raise NotImplementedError("CIFAR100 dataset not available")
        def get_glue_sst2(*args, **kwargs):
            raise NotImplementedError("GLUE SST-2 dataset not available")

def get_dataset(dataset_name, **kwargs):
    """Get dataset loaders by name."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == "cifar10":
        return get_cifar10(**kwargs)
    elif dataset_name == "cifar100":
        return get_cifar100(**kwargs)
    elif dataset_name == "glue-sst2":
        return get_glue_sst2(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

__all__ = ["get_dataset", "get_cifar10", "get_cifar100", "get_glue_sst2"]