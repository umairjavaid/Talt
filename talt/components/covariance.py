"""Covariance estimation component for TALT optimizer."""

import torch

class IncrementalCovariance:
    """
    Computes covariance matrix incrementally to save memory.

    Allows for updating the covariance estimate one sample at a time,
    with exponential forgetting to give more weight to recent samples.
    """

    def __init__(self, dim: int, decay: float = 0.95):
        """
        Initialize incremental covariance estimator.

        Args:
            dim: Dimension of data
            decay: Decay factor for old observations (0-1)
        """
        self.dim = dim
        self.decay = decay
        self.n_samples = 0
        self.mean = torch.zeros(dim)
        self.cov = torch.zeros(dim, dim)

    def update(self, x: torch.Tensor) -> None:
        """
        Update covariance estimate with new data.

        Args:
            x: New data point(s) of shape (batch_size, dim) or (dim,)
        """
        # Ensure input is on CPU and 2D
        x = x.cpu()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        for i in range(batch_size):
            x_i = x[i]

            # Apply decay to previous statistics
            if self.n_samples > 0:
                self.mean *= self.decay
                self.cov *= self.decay**2

            # Update sample count with decay
            self.n_samples = self.decay * self.n_samples + 1

            # Update mean
            delta = x_i - self.mean
            self.mean += delta / self.n_samples

            # Update covariance
            delta2 = x_i - self.mean
            factor = 1.0 / max(1, self.n_samples - 1)
            self.cov += torch.outer(delta, delta2) * factor

    def get_covariance(self, reg: float = 1e-6) -> torch.Tensor:
        """
        Get current covariance estimate with regularization and proper device handling.
        
        Args:
            reg: Regularization parameter to ensure positive definiteness
            
        Returns:
            Regularized covariance matrix on the correct device
        """
        # Determine the device from existing tensors
        device = self.cov.device if hasattr(self.cov, 'device') else 'cpu'
        
        if self.n_samples < 2:
            # Not enough samples, return regularized identity matrix
            return torch.eye(self.dim, device=device) * reg
        
        # Add regularization with correct device
        reg_matrix = torch.eye(self.dim, device=device) * reg
        cov = self.cov + reg_matrix
        
        # Ensure symmetry (important for numerical stability)
        cov = 0.5 * (cov + cov.t())
        
        # Additional numerical stability check
        try:
            # Check if matrix is positive definite
            torch.linalg.cholesky(cov)
        except RuntimeError:
            # If not positive definite, add more regularization
            additional_reg = torch.eye(self.dim, device=device) * (reg * 10)
            cov = cov + additional_reg
        
        return cov
