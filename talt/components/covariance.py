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
        Get current covariance estimate with regularization.

        Args:
            reg: Regularization parameter

        Returns:
            Covariance matrix with regularization
        """
        # Get device of current cov matrix if it exists
        device = self.cov.device if hasattr(self.cov, 'device') else 'cpu'
        
        if self.n_samples < 2:
            # Not enough samples, return identity matrix
            return torch.eye(self.dim, device=device) * reg  # Fixed: Match device with cov

        # Add regularization
        cov = self.cov + torch.eye(self.dim, device=device) * reg  # Fixed: Match device with cov

        # Ensure symmetry
        cov = 0.5 * (cov + cov.t())

        return cov
