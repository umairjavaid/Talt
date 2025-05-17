"""Dimensionality reduction component for TALT optimizer."""

import math
import torch

class RandomProjection:
    """
    Implements sparse random projection for dimension reduction.

    Uses a memory-efficient sparse random projection matrix to reduce
    the dimensionality of input data while approximately preserving
    distances between points.
    """

    def __init__(self, original_dim: int, target_dim: int, seed: int = 42):
        """
        Initialize random projection matrix.

        Args:
            original_dim: Original dimension
            target_dim: Target dimension after projection
            seed: Random seed for reproducibility
        """
        self.original_dim = original_dim
        self.target_dim = min(target_dim, original_dim)

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Create sparse random projection matrix
        sparsity = 1.0 / math.sqrt(self.original_dim)
        self.projection = self._create_sparse_projection(sparsity)

    def _create_sparse_projection(self, sparsity: float) -> torch.Tensor:
        """
        Create a sparse random projection matrix.

        Uses the 'very sparse' random projection method where most values
        are 0, and non-zero values are +/- sqrt(3) with equal probability.

        Args:
            sparsity: Sparsity level (fraction of non-zero elements)

        Returns:
            Sparse projection matrix
        """
        # Create a mask for non-zero elements
        mask = torch.rand(self.target_dim, self.original_dim) < sparsity

        # Create random signs: 1 or -1
        signs = torch.randint(0, 2, (self.target_dim, self.original_dim)) * 2 - 1

        # Scale factor for unit variance
        scale = math.sqrt(1.0 / sparsity)

        # Create projection matrix
        projection = (mask.float() * signs.float() * scale) / math.sqrt(self.target_dim)
        return projection

    def project(self, data: torch.Tensor) -> torch.Tensor:
        """
        Project data from original dimension to target dimension.

        Args:
            data: Data tensor of shape (batch_size, original_dim) or (original_dim,)

        Returns:
            Projected data of shape (batch_size, target_dim) or (target_dim,)
        """
        # Check if data is 1D or 2D
        is_1d = data.dim() == 1

        # Ensure data is 2D for matrix multiplication
        if is_1d:
            data = data.unsqueeze(0)

        # Move projection matrix to same device as data
        projection = self.projection.to(data.device)

        # Project data
        projected = torch.matmul(data, projection.t())

        # Return in original shape
        return projected.squeeze(0) if is_1d else projected
