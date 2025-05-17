"""Valley detection component for TALT optimizer."""

import torch
from collections import deque
from typing import Tuple, Optional

class ValleyDetector:
    """
    Non-parametric valley detection using gradient consistency.

    Analyzes gradient direction changes to identify valleys in the
    loss landscape without relying on eigendecomposition.
    """

    def __init__(self, window_size: int = 5, threshold: float = 0.2):
        """
        Initialize valley detector.

        Args:
            window_size: Size of window for gradient analysis
            threshold: Threshold for valley detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.grad_history = deque(maxlen=window_size)

    def update(self, grad: torch.Tensor) -> None:
        """
        Update gradient history.

        Args:
            grad: Current gradient
        """
        # Store normalized gradient
        grad_norm = grad / (torch.norm(grad) + 1e-10)
        self.grad_history.append(grad_norm.cpu())

    def detect_valley(self) -> Tuple[bool, Optional[torch.Tensor]]:
        """
        Detect if current point is in a valley.

        Returns:
            Tuple of (is_valley, valley_direction)
        """
        if len(self.grad_history) < self.window_size:
            return False, None

        # Convert history to tensor
        grads = torch.stack(list(self.grad_history))

        # Compute gradient consistency (average cosine similarity)
        n = len(self.grad_history)
        cosine_sum = 0.0
        count = 0

        for i in range(n):
            for j in range(i+1, n):
                cos = torch.dot(grads[i], grads[j])
                cosine_sum += cos.item()
                count += 1

        avg_cosine = cosine_sum / max(1, count)

        # If gradients are inconsistent (pointing in different directions)
        # then we might be in a valley
        is_valley = avg_cosine < self.threshold

        # If in valley, compute valley direction using PCA
        if is_valley:
            try:
                # Center gradients
                centered = grads - grads.mean(dim=0, keepdim=True)

                # Compute covariance
                cov = torch.matmul(centered.t(), centered) / (n - 1)

                # Get eigenvector with smallest eigenvalue (valley direction)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                valley_dir = eigenvectors[:, 0]  # Direction of smallest eigenvalue
                return True, valley_dir
            except Exception:
                return is_valley, None

        return is_valley, None
