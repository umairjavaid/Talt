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
        # Calculate the number of non-zero elements per row
        nnz_per_row = max(1, int(sparsity * self.original_dim))

        # Indices for non-zero elements
        indices = []
        values = []

        for row in range(self.target_dim):
            # Randomly select indices for non-zero elements in this row
            row_indices = torch.randperm(self.original_dim)[:nnz_per_row]
            indices.append(torch.stack([torch.full_like(row_indices, row), row_indices]))

            # Assign random signs (+1 or -1) to the non-zero elements
            row_values = torch.randint(0, 2, (nnz_per_row,)) * 2 - 1
            values.append(row_values.float())

        # Combine indices and values
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values)

        # Create sparse tensor
        projection = torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.target_dim, self.original_dim),
            dtype=torch.float32
        )

        # Scale for unit variance
        projection = projection * math.sqrt(1.0 / sparsity) / math.sqrt(self.target_dim)
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

        # Perform sparse matrix multiplication
        projected = torch.sparse.mm(projection, data.t()).t()

        # Return in original shape
        return projected.squeeze(0) if is_1d else projected
