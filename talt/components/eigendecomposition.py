"""Eigendecomposition component for TALT optimizer with numerical stability."""

import torch
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class PowerIteration:
    """
    Computes eigendecomposition using power iteration with numerical stability.
    
    Uses deflation to compute multiple eigenvectors sequentially.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        """
        Initialize power iteration solver.

        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol

    def compute_eigenpairs(self, matrix: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top k eigenvectors and eigenvalues with improved numerical stability.
        
        Args:
            matrix: Input matrix for eigendecomposition
            k: Number of eigenpairs to compute
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Numerical stability constants
        eps = 1e-10
        condition_threshold = 1e12
        
        # Ensure matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for eigendecomposition")
        
        n = matrix.shape[0]
        device = matrix.device
        
        # Check matrix condition number for stability
        try:
            # Estimate condition number using norms
            matrix_norm = torch.norm(matrix, p=2)
            matrix_inv_norm = torch.norm(torch.linalg.pinv(matrix), p=2)
            condition_number = matrix_norm * matrix_inv_norm
            
            if condition_number > condition_threshold:
                logger.warning(f"Matrix is ill-conditioned (condition number: {condition_number:.2e})")
                # Add regularization
                reg = eps * torch.eye(matrix.shape[0], device=matrix.device)
                matrix = matrix + reg
        except Exception:
            # If condition number estimation fails, add small regularization anyway
            reg = eps * torch.eye(matrix.shape[0], device=matrix.device)
            matrix = matrix + reg
        
        # Storage for results
        eigenvalues = torch.zeros(k, device=device)
        eigenvectors = torch.zeros(n, k, device=device)
        
        # Work on a copy to avoid modifying the original
        A = matrix.clone()
        
        for i in range(k):
            # Initialize random vector
            v = torch.randn(n, device=device, dtype=matrix.dtype)
            v = v / torch.norm(v)
            
            # Power iteration
            for iteration in range(self.max_iter):
                v_old = v.clone()
                
                # Matrix-vector multiplication
                v = torch.mv(A, v)
                
                # Normalize with stability check
                norm = torch.norm(v)
                if norm < eps:
                    # If vector collapsed to near-zero, reinitialize
                    v = torch.randn(n, device=device, dtype=matrix.dtype)
                    v = v / torch.norm(v)
                    logger.debug(f"Reinitializing eigenvector {i} due to numerical collapse")
                    continue
                else:
                    v = v / norm
                
                # Check for NaN/Inf
                if torch.isnan(v).any() or torch.isinf(v).any():
                    logger.warning(f"NaN/Inf detected in eigenvector {i}, reinitializing")
                    v = torch.randn(n, device=device, dtype=matrix.dtype)
                    v = v / torch.norm(v)
                    continue
                
                # Check convergence
                diff = torch.norm(v - v_old)
                if diff < self.tol:
                    break
            
            # Compute eigenvalue using Rayleigh quotient
            eigenval = torch.dot(v, torch.mv(A, v))
            
            # Store results
            eigenvalues[i] = eigenval
            eigenvectors[:, i] = v
            
            # Deflation: remove the found eigenvector from the matrix
            # A = A - eigenval * outer(v, v)
            A = A - eigenval * torch.outer(v, v)
            
            # Additional numerical stability: ensure matrix remains symmetric
            A = 0.5 * (A + A.t())
        
        return eigenvalues, eigenvectors

    def compute_dominant_eigenpair(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute only the dominant eigenpair for efficiency.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (dominant_eigenvalue, dominant_eigenvector)
        """
        eigenvals, eigenvecs = self.compute_eigenpairs(matrix, k=1)
        return eigenvals[0], eigenvecs[:, 0]
