"""Components module for the TALT package."""

from talt.components.dimensionality_reduction import RandomProjection
from talt.components.covariance import IncrementalCovariance
from talt.components.eigendecomposition import PowerIteration
from talt.components.valley_detection import ValleyDetector

__all__ = [
    'RandomProjection',
    'IncrementalCovariance',
    'PowerIteration',
    'ValleyDetector'
]
