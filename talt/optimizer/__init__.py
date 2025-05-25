"""Optimizer module for the TALT package."""

from .improved_talt import ImprovedTALTOptimizer
try:
    from .original_talt import TALTOptimizer
except ImportError:
    TALTOptimizer = None

__all__ = ['ImprovedTALTOptimizer']
if TALTOptimizer is not None:
    __all__.append('TALTOptimizer')
