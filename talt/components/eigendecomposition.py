"""Eigendecomposition component for TALT optimizer."""

import torch
from typing import Tuple

class PowerIteration:
    """
    Computes top eigenvectors using power iteration method.

    A more stable alternative to direct eigendecomposition that
    iteratively finds principal eigenvectors and eigenvalues.
    """

    def __init__(self, max_iter: int = 20, tol: float = 1e-6):
        """
        Initialize power iteration.

        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol

    def compute_eigenpairs(self, matrix: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top k eigenvectors and eigenvalues.

        Args:
            matrix: Square matrix
            k: Number of eigenpairs to compute

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Create a copy to avoid modifying the original matrix
        matrix = matrix.clone()
        
        n = matrix.shape[0]
        k = min(k, n)

        # Initialize storage for eigenvectors and eigenvalues
        eigenvectors = torch.zeros((n, k), device=matrix.device)
        eigenvalues = torch.zeros(k, device=matrix.device)

        # Initial random vectors, orthogonalized
        vecs = torch.randn(n, k, device=matrix.device)
        vecs, _ = torch.linalg.qr(vecs)

        # Deflation approach: find eigenvectors one by one
        for i in range(k):
            # Current vector
            v = vecs[:, i].clone()

            # Power iteration
            for _ in range(self.max_iter):
                prev_v = v.clone()

                # Apply matrix
                v = torch.mv(matrix, v)

                # Orthogonalize against previous eigenvectors
                for j in range(i):
                    v = v - torch.dot(v, eigenvectors[:, j]) * eigenvectors[:, j]

                # Normalize
                norm = torch.norm(v)
                if norm > 1e-10:
                    v = v / norm
                else:
                    # If vector is close to zero, reset with random
                    v = torch.randn_like(v)
                    v = v / torch.norm(v)

                # Check convergence
                cosine = torch.abs(torch.dot(v, prev_v))
                if cosine > 1 - self.tol:
                    break

            # Compute Rayleigh quotient for eigenvalue
            eigenvalues[i] = torch.dot(v, torch.mv(matrix, v))
            eigenvectors[:, i] = v

            # Deflate matrix to find next eigenvector
            matrix = matrix - eigenvalues[i] * torch.outer(v, v)

        return eigenvalues, eigenvectors
