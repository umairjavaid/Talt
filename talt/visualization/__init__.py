"""
TALT Visualization Package

This package provides visualization tools for the TALT optimizer, including:
- ImprovedTALTVisualizer: Modern visualization for the improved TALT optimizer
- TALTVisualizer: Visualization for the original TALT optimizer
"""

import os
import sys
from pathlib import Path

# Ensure the talt package is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import core visualization components
try:
    from talt.visualization.visualizer import ImprovedTALTVisualizer
    from talt.visualization.original_visualizer import TALTVisualizer
except ImportError as e:
    # Provide fallback imports for partial installations
    import warnings
    warnings.warn(f"Error importing visualization components: {e}")
    
    # Define placeholder classes if imports fail
    class ImprovedTALTVisualizer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ImprovedTALTVisualizer could not be imported")
    
    class TALTVisualizer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("TALTVisualizer could not be imported")

# Define the public API
__all__ = ['ImprovedTALTVisualizer', 'TALTVisualizer']
