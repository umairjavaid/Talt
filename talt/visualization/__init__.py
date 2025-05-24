"""
TALT Visualization Package

This package provides visualization tools for the TALT optimizer, including:
- ImprovedTALTVisualizer: Modern visualization for the improved TALT optimizer
- OriginalTALTVisualizer: Visualization for the original TALT optimizer
"""

from .visualizer import TALTVisualizer

# Try to import the original visualizer
try:
    from .original_visualizer import OriginalTALTVisualizer
    ORIGINAL_VIZ_AVAILABLE = True
except ImportError:
    OriginalTALTVisualizer = None
    ORIGINAL_VIZ_AVAILABLE = False

# Define the public API
__all__ = ['TALTVisualizer', 'OriginalTALTVisualizer']
