#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file can be left empty or used to mark the directory as a package.
# If you have specific sub-packages or modules to expose from talt_evaluation,
# you can import them here.

# Example:
# from .analysis import results_aggregator
# from .datasets import cifar, glue
# from .experiments import experiment
# from .models import get_architecture, BaseArchitecture

# For now, keeping it simple as the main functionality is likely accessed
# through specific scripts or modules directly.

# If run_experiment.py or other top-level scripts in talt_evaluation
# rely on specific imports from submodules being available directly under
# `talt_evaluation.X`, then those should be listed here.
# Otherwise, direct imports like `from talt_evaluation.models import get_architecture`
# will work fine without changes here.

# Considering the refactoring, we might want to expose the new models entry point:
from .models import get_architecture
from .datasets import cifar, glue  # Assuming these are still relevant entry points
from .experiments import experiment  # Assuming this is a relevant entry point

__all__ = ["get_architecture", "cifar", "glue", "experiment"]