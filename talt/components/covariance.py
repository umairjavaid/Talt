"""Incremental covariance estimation for TALT optimizer."""

import torch
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

class IncrementalCovariance:
    """
    Incremental covariance matrix estimation with exponential decay.
    
    This class maintains a running estimate of the covariance matrix
    of input vectors using exponential moving averages.
    """
    
    def __init__(
        self, 
        dim: int, 
        decay: float = 0.95, 
        reg: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize incremental covariance estimator.
        
        Args:
            dim: Dimension of input vectors
            decay: Exponential decay factor (closer to 1 = longer memory)
            reg: Regularization to add to diagonal
            device: Device to store matrices on
        """
        self.dim = dim
        self.decay = decay
        self.reg = reg
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize on specified device
        self.mean = torch.zeros(dim, device=self.device)
        self.cov = torch.eye(dim, device=self.device) * reg
        self.count = 0
        
    def update(self, x: torch.Tensor) -> None:
        """
        Update covariance estimate with new vector.
        
        Args:
            x: Input vector of shape (dim,)
        """
        # Move to CPU for computation, then back to target device
        x = x.detach().cpu()
        
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected vector of dim {self.dim}, got {x.shape[0]}")
        
        self.count += 1
        
        if self.count == 1:
            # First update - move to target device
            self.mean = x.to(self.device)
            self.cov = torch.eye(self.dim, device=self.device) * self.reg
        else:
            # Move current state to CPU for computation
            mean_cpu = self.mean.cpu()
            cov_cpu = self.cov.cpu()
            
            # Incremental mean update
            delta = x - mean_cpu
            new_mean = mean_cpu + delta / self.count
            
            # Incremental covariance update with decay
            delta2 = x - new_mean
            new_cov = (
                self.decay * cov_cpu + 
                (1 - self.decay) * torch.outer(delta, delta2)
            )
            
            # Move back to target device
            self.mean = new_mean.to(self.device)
            self.cov = new_cov.to(self.device)
    
    def get_covariance(self) -> torch.Tensor:
        """
        Get current covariance estimate with regularization.
        
        Returns:
            Covariance matrix of shape (dim, dim) on target device
        """
        # Ensure regularization and return on correct device
        regularized_cov = self.cov + self.reg * torch.eye(self.dim, device=self.device)
        return regularized_cov
    
    def reset(self) -> None:
        """Reset the covariance estimator."""
        self.mean = torch.zeros(self.dim, device=self.device)
        self.cov = torch.eye(self.dim, device=self.device) * self.reg
        self.count = 0
