"""
Improved Topology-Aware Learning Trajectory (TALT) Optimizer

This package implements an enhanced version of the TALT optimizer that uses
dimensionality reduction, robust eigendecomposition, and non-parametric
valley detection to improve optimization performance while reducing memory usage.
"""

from talt.utils import Timer, print_memory_usage, PerformanceTracker
from talt.components import RandomProjection, IncrementalCovariance, PowerIteration, ValleyDetector
from .optimizer import ImprovedTALTOptimizer
try:
    from .optimizer import TALTOptimizer
except ImportError:
    TALTOptimizer = None
from talt.visualization import TALTVisualizer, OriginalTALTVisualizer

# Export diagnostic utilities for TALT optimizers with theoretical fixes
def diagnose_talt_optimizer(optimizer):
    """Utility function to diagnose TALT optimizer state with theoretical fixes analysis."""
    if hasattr(optimizer, 'diagnose_visualization_state'):
        optimizer.diagnose_visualization_state()
    elif hasattr(optimizer, 'diagnose_convergence'):
        optimizer.diagnose_convergence()
        
        # Additional diagnostics for theoretical fixes
        if hasattr(optimizer, 'use_adaptive_memory'):
            print(f"\nTheoretical Fixes Status:")
            print(f"- Adaptive Memory: {optimizer.use_adaptive_memory}")
            print(f"- Gradient Smoothing: {getattr(optimizer, 'use_gradient_smoothing', False)}")
            print(f"- Adaptive Thresholds: {getattr(optimizer, 'use_adaptive_thresholds', False)}")
            print(f"- Parameter Normalization: {getattr(optimizer, 'use_parameter_normalization', False)}")
            print(f"- Incremental Covariance: {getattr(optimizer, 'use_incremental_covariance', False)}")
    else:
        print(f"Optimizer {type(optimizer).__name__} does not support diagnostics")

def force_talt_update(optimizer):
    """Utility function to force TALT topology update."""
    if hasattr(optimizer, 'force_topology_update'):
        optimizer.force_topology_update()
        print("TALT topology update forced")
    else:
        print(f"Optimizer {type(optimizer).__name__} does not support forced updates")

def create_enhanced_talt_config():
    """Create a configuration dict with optimal theoretical fixes settings."""
    return {
        'lr': 0.01,
        'memory_size': 25,  # Will be adaptive per parameter
        'update_interval': 15,
        'valley_strength': 0.15,
        'smoothing_factor': 0.4,
        'grad_store_interval': 3,
        'min_param_size': 50,
        
        # Theoretical fixes (optimal settings)
        'use_adaptive_memory': True,
        'use_gradient_smoothing': True,
        'smoothing_beta': 0.9,
        'use_adaptive_thresholds': True,
        'use_parameter_normalization': True,
        'use_incremental_covariance': True,
        'eigenspace_blend_factor': 0.7,
        'min_memory_ratio': 0.1,
    }

__all__ = [
    'Timer',
    'print_memory_usage',
    'PerformanceTracker',
    'RandomProjection',
    'IncrementalCovariance',
    'PowerIteration',
    'ValleyDetector',
    'ImprovedTALTOptimizer',
    'TALTVisualizer',
    'OriginalTALTVisualizer',
    'diagnose_talt_optimizer',
    'force_talt_update',
    'create_enhanced_talt_config'  # New export
]

if TALTOptimizer is not None:
    __all__.append('TALTOptimizer')
