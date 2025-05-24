"""
Improved Topology-Aware Learning Trajectory (TALT) Optimizer

This package implements an enhanced version of the TALT optimizer that uses
dimensionality reduction, robust eigendecomposition, and non-parametric
valley detection to improve optimization performance while reducing memory usage.
"""

from talt.utils import Timer, print_memory_usage, PerformanceTracker
from talt.components import RandomProjection, IncrementalCovariance, PowerIteration, ValleyDetector
from talt.optimizer import ImprovedTALTOptimizer, TALTOptimizer
from talt.visualization import TALTVisualizer, OriginalTALTVisualizer

# Export diagnostic utilities for TALT optimizers
def diagnose_talt_optimizer(optimizer):
    """Utility function to diagnose TALT optimizer state."""
    if hasattr(optimizer, 'diagnose_visualization_state'):
        optimizer.diagnose_visualization_state()
    else:
        print(f"Optimizer {type(optimizer).__name__} does not support diagnostics")

def force_talt_update(optimizer):
    """Utility function to force TALT topology update."""
    if hasattr(optimizer, 'force_topology_update'):
        optimizer.force_topology_update()
        print("TALT topology update forced")
    else:
        print(f"Optimizer {type(optimizer).__name__} does not support forced updates")

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
    'TALTVisualizer',
    'OriginalTALTVisualizer',
    'diagnose_talt_optimizer',
    'force_talt_update'
]
