#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Add current directory to path for relative imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import modules with error handling
try:
    from .models import get_architecture
    from .datasets import get_dataset
    from .experiments import Experiment
    from .hyperparameter_tuning import TaltTuner
    from .visualization import create_training_report
except ImportError:
    # Fallback for direct script execution
    try:
        from models import get_architecture
        from datasets import get_dataset  
        from experiments import Experiment
        from hyperparameter_tuning import TaltTuner
        from visualization import create_training_report
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import talt_evaluation modules: {e}")

__all__ = ["get_architecture", "get_dataset", "Experiment", "TaltTuner", "create_training_report"]