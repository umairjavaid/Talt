"""Main implementation of the Improved TALT Optimizer."""

import os
import time
import gc
import torch
import torch.nn as nn
import logging
import psutil
import numpy as np
from collections import deque
from torch.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

from talt.components import (
    RandomProjection,
    IncrementalCovariance,
    PowerIteration,
    ValleyDetector
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedTALTOptimizer:
    """
    Improved Topology-Aware Learning Trajectory Optimizer

    This optimizer implements an enhanced version of TALT with:
    1. Dimension reduction via random projections
    2. Incremental covariance estimation
    3. Robust eigendecomposition via power iteration
    4. Non-parametric valley detection
    """
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: Callable,
        *,
        lr: float = 1e-2,
        projection_dim: int = 32,
        memory_size: int = 10,
        update_interval: int = 20,
        valley_strength: float = 0.2,
        smoothing_factor: float = 0.3,
        grad_store_interval: int = 5,
        min_param_size: int = 100,
        cov_decay: float = 0.95,
        adaptive_reg: bool = True,
        device: Union[str, torch.device] = "cuda",
        max_stored_steps: int = 1000,
        max_visualization_points: int = 100
    ):
        """
        Initialize the improved TALT optimizer.

        Args:
            model: Neural network model
            base_optimizer: Base optimizer class (e.g., optim.SGD)
            lr: Learning rate
            projection_dim: Dimension after random projection
            memory_size: Number of past gradients to store
            update_interval: Steps between topology updates
            valley_strength: Strength of valley acceleration
            smoothing_factor: Factor for smoothing high-curvature directions
            grad_store_interval: Steps between gradient storage
            min_param_size: Minimum parameter size to track
            cov_decay: Decay factor for incremental covariance
            adaptive_reg: Whether to use adaptive regularization
            device: Device to perform computations on
            max_stored_steps: Maximum steps to store in history
            max_visualization_points: Maximum visualization datapoints
        """
        self.model = model
        self.optimizer = base_optimizer(model.parameters(), lr=lr)
        self.projection_dim = projection_dim
        self.memory_size = memory_size
        self.update_interval = update_interval
        self.valley_strength = valley_strength
        self.smoothing_factor = smoothing_factor
        self.store_interval = grad_store_interval
        self.min_param_size = min_param_size
        self.cov_decay = cov_decay
        self.adaptive_reg = adaptive_reg
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.max_stored_steps = max_stored_steps
        self.max_visualization_points = max_visualization_points

        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler('cuda')

        # Tracking variables
        self.steps = 0
        self.loss_history = deque(maxlen=max_stored_steps)
        self.bifurcations = deque(maxlen=100)  # Track bifurcation points

        # Thread pool for asynchronous operations
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._topology_update_future = None

        # Visualization data with limited history
        self._visualization_data = {
            'loss_values': deque(maxlen=max_visualization_points),
            'valley_detections': deque(maxlen=max_visualization_points),
            'gradient_stats': {}
        }

        # Parameter-specific structures
        self.param_data = {}

        # Register hooks for parameters of sufficient size
        for name, p in model.named_parameters():
            if p.requires_grad and p.numel() > self.min_param_size:
                # Create dimension-reduced representation
                dim = p.numel()
                target_dim = min(self.projection_dim, dim // 4)  # No more than 1/4 of original

                self.param_data[name] = {
                    'projector': RandomProjection(dim, target_dim),
                    'covariance': IncrementalCovariance(target_dim, decay=cov_decay),
                    'valley_detector': ValleyDetector(window_size=5, threshold=0.2),
                    'valley_dirs': None,
                    'transformation': None,
                    'gradient_norm_history': deque(maxlen=50),
                    'consistency_history': deque(maxlen=50)
                }

                self._visualization_data['gradient_stats'][name] = deque(maxlen=max_visualization_points)

                # Register gradient hook
                p.register_hook(lambda grad, name=name: self._transform_gradient(grad, name))

    def _transform_gradient(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """
        Transform gradient using learned topology information.

        Args:
            grad: Original gradient
            name: Parameter name

        Returns:
            Transformed gradient
        """
        if name not in self.param_data:
            return grad

        # Make a copy of original gradient
        orig_grad = grad.clone()
        flat_grad = grad.view(-1)

        # Get parameter data
        param_info = self.param_data[name]
        transformation = param_info['transformation']

        # If no transformation available, just return original
        if transformation is None:
            return grad

        try:
            # Update valley detector with current gradient
            param_info['valley_detector'].update(flat_grad.detach().cpu())

            # Store gradient norm
            grad_norm = torch.norm(flat_grad).item()
            param_info['gradient_norm_history'].append(grad_norm)

            # Apply learned transformation
            is_valley, valley_dir = param_info['valley_detector'].detect_valley()

            if is_valley and valley_dir is not None:
                # Record bifurcation point
                if len(self.bifurcations) < self.max_stored_steps:
                    self.bifurcations.append(self.steps)

                # Store detection for visualization
                self._visualization_data['valley_detections'].append(
                    (self.steps, name, 'valley')
                )

                # Project valley direction to original space if needed
                if valley_dir.shape[0] != flat_grad.shape[0]:
                    # This would happen if we're using dimension reduction
                    # We need an approximate mapping back to original space
                    # For now, we just rely on the transformation matrix
                    pass

                # Transform the gradient
                # First, compute the component in valley direction
                valley_dir = valley_dir.to(self.device)
                valley_component = torch.dot(flat_grad, valley_dir) * valley_dir

                # Amplify the valley component
                flat_grad = flat_grad + self.valley_strength * valley_component

            # Apply curvature-based transformation
            transformed_grad = torch.matmul(transformation, flat_grad.unsqueeze(1)).squeeze(1)

            # Check if transformation is reasonable
            cos_sim = nn.functional.cosine_similarity(
                transformed_grad.view(-1), orig_grad.view(-1), dim=0
            )

            # If very different from original, blend back
            if cos_sim < 0.6:
                blend_factor = 0.6 - cos_sim
                transformed_grad = (1.0 - blend_factor) * transformed_grad.view_as(orig_grad) + blend_factor * orig_grad
            else:
                transformed_grad = transformed_grad.view_as(orig_grad)

            # Safety check for NaN or Inf
            if torch.isnan(transformed_grad).any() or torch.isinf(transformed_grad).any():
                logger.warning(f"NaN or Inf in transformed gradient for {name}")
                return orig_grad

            return transformed_grad

        except Exception as e:
            logger.warning(f"Error transforming gradient for {name}: {e}")
            return orig_grad

    def _update_topology(self) -> None:
        """Update topology information for all tracked parameters."""
        for name, param_info in self.param_data.items():
            # Skip if not enough data
            if len(param_info['gradient_norm_history']) < 3:
                continue

            try:
                # Get the covariance matrix
                cov = param_info['covariance'].get_covariance()

                # Adaptive regularization based on condition number
                if self.adaptive_reg:
                    try:
                        eigs = torch.linalg.eigvalsh(cov)
                        if eigs[0] > 0:  # Avoid division by zero
                            condition_number = eigs[-1] / eigs[0]
                            reg = max(1e-6, min(1e-2, 1e-5 * condition_number))
                            cov = cov + reg * torch.eye(cov.shape[0], device=cov.device)
                    except Exception:
                        # Fallback: add standard regularization
                        cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)

                # Use power iteration for more stable eigendecomposition
                power_iter = PowerIteration(max_iter=20, tol=1e-5)
                eigenvalues, eigenvectors = power_iter.compute_eigenpairs(cov, k=min(5, cov.shape[0]))

                # Create transformation matrix for gradient adjustment
                # This is like a preconditioner based on the eigenstructure
                transform = torch.eye(cov.shape[0], device=cov.device)

                for i, val in enumerate(eigenvalues):
                    vec = eigenvectors[:, i].unsqueeze(1)
                    abs_val = abs(val.item())

                    if abs_val > 1.0:
                        # Reduce step size in high-curvature directions
                        scale = 1.0 / np.sqrt(abs_val) * self.smoothing_factor
                    elif abs_val < 0.2:
                        # Boost step size in flat regions
                        scale = 1.5
                    else:
                        scale = 1.0

                    transform += (scale - 1.0) * torch.matmul(vec, vec.t())

                # Store transformation matrix
                param_info['transformation'] = transform.to(self.device)

                # Store top eigenvalues for visualization if needed
                top_vals = eigenvalues[:min(3, len(eigenvalues))].detach().cpu().numpy()
                self._visualization_data['gradient_stats'][name].append({
                    'step': self.steps,
                    'eigenvalues': top_vals,
                    'grad_norm': np.mean([n for n in param_info['gradient_norm_history']])
                })

                # Clean up to avoid memory leaks
                del cov, eigenvalues, eigenvectors, transform

            except Exception as e:
                logger.warning(f"Error updating topology for {name}: {e}")

    def _update_topology_async(self) -> None:
        """Update topology asynchronously."""
        if self._topology_update_future and not self._topology_update_future.done():
            return

        self._topology_update_future = self.executor.submit(self._update_topology)

    def step(self, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Perform an optimization step.

        Args:
            loss_fn: Loss function
            x: Input data
            y: Target data

        Returns:
            Tuple of (loss value, model output)
        """
        # Initialize timings
        timings = {}
        batch_start = time.time()

        self.optimizer.zero_grad()
        self.model.train()

        # Forward pass
        forward_start = time.time()
        with autocast('cuda'):
            out = self.model(x)
            loss = loss_fn(out, y)
        forward_time = time.time() - forward_start
        timings['forward_pass'] = forward_time

        # Backward pass
        backward_start = time.time()
        self.scaler.scale(loss).backward()
        backward_time = time.time() - backward_start
        timings['backward_pass'] = backward_time

        # Store and analyze gradients periodically
        grad_start = time.time()
        if self.steps % self.store_interval == 0:
            for name, p in self.model.named_parameters():
                if p.grad is not None and name in self.param_data:
                    flat_grad = p.grad.detach().view(-1)

                    # Update covariance with projected gradient
                    projected_grad = self.param_data[name]['projector'].project(flat_grad)
                    self.param_data[name]['covariance'].update(projected_grad)
        grad_time = time.time() - grad_start
        timings['gradient_processing'] = grad_time

        # Update parameters
        optim_start = time.time()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        optim_time = time.time() - optim_start
        timings['optimizer_step'] = optim_time

        # Track progress
        self.steps += 1
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self._visualization_data['loss_values'].append(loss_value)

        # Print progress
        if self.steps % 10 == 0 or self.steps == 1:
            # Current memory usage
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                mem_str = f"GPU: {mem_allocated:.1f}MB / {mem_reserved:.1f}MB"
            else:
                process = psutil.Process(os.getpid())
                mem_usage = process.memory_info().rss / (1024 * 1024)  # MB
                mem_str = f"RAM: {mem_usage:.1f}MB"

            print(f"Step {self.steps:4d} | Loss: {loss_value:.6f} | {mem_str} | "
                  f"F: {forward_time:.4f}s, B: {backward_time:.4f}s, O: {optim_time:.4f}s")

        # Update topology information periodically
        topo_time = 0
        if self.steps % self.update_interval == 0:
            topo_start = time.time()
            self._update_topology_async()
            topo_time = time.time() - topo_start
            timings['topology_update'] = topo_time

            # Log topology update
            print(f"🔄 Topology update at step {self.steps} took {topo_time:.4f}s")

        # Periodic cleanup
        if self.steps % 100 == 0:
            cleanup_start = time.time()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_time = time.time() - cleanup_start

            print(f"🧹 Memory cleanup at step {self.steps} took {cleanup_time:.4f}s")

        # Total batch time
        batch_time = time.time() - batch_start
        timings['batch_total'] = batch_time

        return loss_value, out

    def shutdown(self) -> None:
        """Clean up resources."""
        # Cancel any pending tasks
        if self._topology_update_future and not self._topology_update_future.done():
            self._topology_update_future.cancel()

        # Shut down executor
        self.executor.shutdown(wait=False)

        # Clear memory
        self.param_data.clear()
        self._visualization_data.clear()

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
