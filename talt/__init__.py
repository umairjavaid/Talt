"""
Improved Topology-Aware Learning Trajectory (TALT) Optimizer

This package implements an enhanced version of the TALT optimizer that uses
dimensionality reduction, robust eigendecomposition, and non-parametric
valley detection to improve optimization performance while reducing memory usage.
"""

from talt.utils import Timer, print_memory_usage, PerformanceTracker
from talt.components import RandomProjection, IncrementalCovariance, PowerIteration, ValleyDetector
from talt.optimizer import ImprovedTALTOptimizer, TALTOptimizer
from talt.visualization import ImprovedTALTVisualizer, TALTVisualizer
from talt_evaluation.models import SimpleCNN

__all__ = [
    'Timer',
    'print_memory_usage',
    'PerformanceTracker',
    'RandomProjection',
    'IncrementalCovariance',
    'PowerIteration',
    'ValleyDetector',
    'ImprovedTALTOptimizer',
    'TALTOptimizer',
    'ImprovedTALTVisualizer',
    'TALTVisualizer',
    'SimpleCNN'
]
