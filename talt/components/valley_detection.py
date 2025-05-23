"""Valley detection component for TALT optimizer with numerical stability."""

import torch
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ValleyDetector:
    """
    Detects valleys in the loss landscape using gradient history analysis.
    
    A valley is characterized by consistent gradient directions and
    potentially oscillating behavior in the loss.
    """

    def __init__(self, history_size: int = 10, valley_threshold: float = 0.1, window_size: int = 5, threshold: Optional[float] = None):
        """
        Initialize valley detector.

        Args:
            history_size: Number of gradients to keep in history
            valley_threshold: Threshold for valley detection
            window_size: Size of the window for gradient consistency checks
            threshold: Additional threshold parameter for compatibility
        """
        self.history_size = history_size
        self.valley_threshold = valley_threshold
        self.window_size = window_size
        self.threshold = threshold  # Store the threshold if provided
        self.grad_history: List[torch.Tensor] = []

    def update(self, grad: torch.Tensor) -> None:
        """
        Update gradient history with numerical stability checks.
        
        Args:
            grad: Current gradient tensor
        """
        # Numerical stability constants
        eps = 1e-8
        min_grad_norm = 1e-10
        
        # Compute gradient norm
        grad_norm_val = torch.norm(grad)
        
        # Skip near-zero gradients to avoid numerical issues
        if grad_norm_val < min_grad_norm:
            logger.debug(f"Skipping near-zero gradient with norm {grad_norm_val}")
            return
        
        # Normalize gradient with epsilon for stability
        grad_norm = grad / (grad_norm_val + eps)
        
        # Additional check for NaN/Inf
        if torch.isnan(grad_norm).any() or torch.isinf(grad_norm).any():
            logger.warning("NaN or Inf detected in normalized gradient, skipping update")
            return
        
        # Store normalized gradient on CPU to save GPU memory
        self.grad_history.append(grad_norm.detach().cpu())
        
        # Maintain history size
        if len(self.grad_history) > self.history_size:
            self.grad_history.pop(0)

    def is_in_valley(self) -> bool:
        """
        Determine if we're currently in a valley based on gradient history.
        
        Returns:
            True if valley is detected, False otherwise
        """
        if len(self.grad_history) < 3:
            return False
        
        try:
            # Compute gradient consistency (dot products between consecutive gradients)
            consistencies = []
            for i in range(len(self.grad_history) - 1):
                dot_product = torch.dot(self.grad_history[i].flatten(), 
                                      self.grad_history[i + 1].flatten())
                consistencies.append(dot_product.item())
            
            # Valley detected if gradients are consistently similar
            avg_consistency = sum(consistencies) / len(consistencies)
            return avg_consistency > self.valley_threshold
            
        except Exception as e:
            logger.warning(f"Error in valley detection: {e}")
            return False

    def get_valley_direction(self) -> Optional[torch.Tensor]:
        """
        Get the dominant direction in the valley.
        
        Returns:
            Average gradient direction if in valley, None otherwise
        """
        if not self.is_in_valley() or len(self.grad_history) == 0:
            return None
        
        try:
            # Compute average gradient direction
            avg_grad = torch.zeros_like(self.grad_history[0])
            for grad in self.grad_history:
                avg_grad += grad
            
            avg_grad = avg_grad / len(self.grad_history)
            
            # Normalize the average direction
            norm = torch.norm(avg_grad)
            if norm > 1e-10:
                return avg_grad / norm
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error computing valley direction: {e}")
            return None
