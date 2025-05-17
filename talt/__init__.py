"""
Improved Topology-Aware Learning Trajectory (TALT) Optimizer

This package implements an enhanced version of the TALT optimizer that uses
dimensionality reduction, robust eigendecomposition, and non-parametric
valley detection to improve optimization performance while reducing memory usage.
"""

from talt.utils import Timer, print_memory_usage, PerformanceTracker
from talt.components import RandomProjection, IncrementalCovariance, PowerIteration, ValleyDetector
from talt.optimizer import ImprovedTALTOptimizer
from talt.model import SimpleCNN
from talt.visualization import ImprovedTALTVisualizer
from talt.train import get_loaders, train_and_evaluate_improved, evaluate

__all__ = [
    'Timer',
    'print_memory_usage',
    'PerformanceTracker',
    'RandomProjection',
    'IncrementalCovariance',
    'PowerIteration',
    'ValleyDetector',
    'ImprovedTALTOptimizer',
    'SimpleCNN',
    'ImprovedTALTVisualizer',
    'get_loaders',
    'train_and_evaluate_improved',
    'evaluate'
]
