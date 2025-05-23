"""
TALT Visualization Package

This package provides visualization tools for the TALT optimizer, including:
- ImprovedTALTVisualizer: Modern visualization for the improved TALT optimizer
- TALTVisualizer: Visualization for the original TALT optimizer
"""

from .visualizer import TALTVisualizer

# Try to import the improved visualizer, fall back to basic if not available
try:
    from .original_visualizer import OriginalTALTVisualizer
    ORIGINAL_VIZ_AVAILABLE = True
except ImportError:
    OriginalTALTVisualizer = None
    ORIGINAL_VIZ_AVAILABLE = False

# Define the public API
__all__ = ['TALTVisualizer']

if ORIGINAL_VIZ_AVAILABLE:
    __all__.append('OriginalTALTVisualizer')
