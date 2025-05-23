"""Main implementation of the Improved TALT Optimizer."""

import os
import time
import gc
import torch
import torch.nn as nn
import logging
import psutil
import numpy as np
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

# Direct imports without version fallbacks
from torch.amp import autocast, GradScaler

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
    
    Production-ready TALT with memory optimizations and robust numerical handling.
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
        max_visualization_points: int = 100,
        grad_clip_norm: Optional[float] = 1.0,
        grad_clip_value: Optional[float] = None,
        detect_anomaly: bool = False
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
        
        # Validate base_optimizer is callable
        if not callable(base_optimizer):
            raise TypeError("base_optimizer must be a callable that returns an optimizer")
        
        # Test base_optimizer call to ensure it works
        try:
            test_optimizer = base_optimizer(model.parameters(), lr=lr)
            if not hasattr(test_optimizer, 'step') or not hasattr(test_optimizer, 'zero_grad'):
                raise TypeError("base_optimizer must return a valid PyTorch optimizer")
            self.optimizer = test_optimizer
        except Exception as e:
            raise ValueError(f"base_optimizer failed to create valid optimizer: {e}")
        
        self.projection_dim = projection_dim
        self.memory_size = memory_size
        self.update_interval = update_interval
        self.valley_strength = min(abs(valley_strength), 1.0)  # Clamp valley strength
        self.smoothing_factor = smoothing_factor
        self.store_interval = grad_store_interval
        self.min_param_size = min_param_size
        self.cov_decay = cov_decay
        self.adaptive_reg = adaptive_reg
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.max_stored_steps = max_stored_steps
        self.max_visualization_points = max_visualization_points
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.detect_anomaly = detect_anomaly

        # Thread safety attributes
        self._state_lock = threading.Lock()
        self._pending_updates = set()
        self._update_in_progress = False
        self.anomaly_count = 0
        self.max_anomalies = 10

        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler()

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
                    'covariance': IncrementalCovariance(target_dim, decay=cov_decay, device=self.device),
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
        Transform gradient using learned topology information with thread safety.

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

        # Check if update is in progress for this parameter
        with self._state_lock:
            if name in self._pending_updates:
                # Skip transformation during update
                return grad

            # Get parameter data safely
            param_info = self.param_data[name]
            transformation = param_info.get('transformation')

        # If no transformation available, return original
        if transformation is None:
            return grad

        try:
            # Apply transformation outside of lock to avoid blocking
            flat_grad = grad.view(-1)

            # Ensure transformation is on correct device
            if transformation is not None and transformation.device != flat_grad.device:
                transformation = transformation.to(flat_grad.device)

            # Update valley detector (thread-safe)
            with self._state_lock:
                param_info['valley_detector'].update(flat_grad.detach().cpu())

                # Store gradient norm
                grad_norm = torch.norm(flat_grad).item()
                param_info['gradient_norm_history'].append(grad_norm)

            # Check for valleys with proper null handling
            is_valley, valley_dir = param_info['valley_detector'].detect_valley()

            if is_valley and valley_dir is not None:
                # Thread-safe bifurcation recording
                with self._state_lock:
                    if len(self.bifurcations) < self.max_stored_steps:
                        self.bifurcations.append(self.steps)

                    self._visualization_data['valley_detections'].append(
                        (self.steps, name, 'valley')
                    )

                # Apply valley transformation with clipping
                valley_dir = valley_dir.to(flat_grad.device)
                valley_component = torch.dot(flat_grad, valley_dir) * valley_dir
                
                # Clip valley component to prevent gradient explosion
                valley_component = torch.clamp(valley_component, -10.0, 10.0)
                
                flat_grad = flat_grad + self.valley_strength * valley_component

            # Apply curvature-based transformation
            transformed_grad = torch.matmul(transformation, flat_grad.unsqueeze(1)).squeeze(1)

            # Safety checks
            cos_sim = nn.functional.cosine_similarity(
                transformed_grad.view(-1), orig_grad.view(-1), dim=0
            )

            if cos_sim < 0.6:
                blend_factor = 0.6 - cos_sim
                transformed_grad = (1.0 - blend_factor) * transformed_grad.view_as(orig_grad) + blend_factor * orig_grad
            else:
                transformed_grad = transformed_grad.view_as(orig_grad)

            # NaN/Inf check
            if torch.isnan(transformed_grad).any() or torch.isinf(transformed_grad).any():
                logger.warning(f"NaN or Inf in transformed gradient for {name}")
                return orig_grad

            return transformed_grad

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU OOM in gradient transformation for {name}: {e}")
            else:
                logger.warning(f"Runtime error in gradient transformation for {name}: {e}")
            return orig_grad
        except ValueError as e:
            logger.warning(f"Value error in gradient transformation for {name}: {e}")
            return orig_grad
        except Exception as e:
            logger.warning(f"Unexpected error transforming gradient for {name}: {e}")
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
                        eigenvals = torch.linalg.eigvalsh(cov)
                        condition_number = torch.abs(eigenvals[-1] / eigenvals[0]) if eigenvals[0] != 0 else 1e12
                        reg = max(1e-6, min(1e-2, 1e-5 * condition_number))
                        cov = cov + reg * torch.eye(cov.shape[0], device=cov.device)
                    except RuntimeError:
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

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU OOM in topology update for {name}: {e}")
                else:
                    logger.warning(f"Runtime error updating topology for {name}: {e}")
            except ValueError as e:
                logger.warning(f"Value error updating topology for {name}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error updating topology for {name}: {e}")

    def step(self, loss_fn: Callable, x: Union[torch.Tensor, Dict[str, torch.Tensor]], 
             y: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """
        Perform an optimization step with gradient clipping and NaN detection.
        
        Args:
            loss_fn: Loss function that takes (predictions, targets) and returns loss
            x: Input data - either a tensor (CNN) or dict (BERT/transformers)
            y: Target data - if None, extracted from x['labels'] for dict inputs
            
        Returns:
            Tuple of (loss_value, model_output)
        """
        # Validate batch structure
        if not isinstance(x, (torch.Tensor, dict)):
            raise TypeError(f"Unsupported batch type: {type(x)}. Expected tensor or dict.")
        
        # Initialize timings
        timings = {}
        batch_start = time.time()
        
        self.optimizer.zero_grad()
        self.model.train()
        
        # Handle different input types
        if isinstance(x, dict):
            # Transformer model with dictionary inputs
            # Extract labels if y not provided
            if y is None:
                y = x.get('labels')
                if y is None:
                    raise ValueError("Dictionary input must contain 'labels' key or y must be provided")
            
            # Separate model inputs from labels
            model_inputs = {k: v.to(self.device) for k, v in x.items() if k != 'labels'}
            y = y.to(self.device)
            
            # Forward pass
            forward_start = time.time()
            with autocast('cuda'):
                out = self.model(**model_inputs)
                
                # Handle different output formats
                if hasattr(out, 'logits'):  # transformers ModelOutput
                    logits = out.logits
                elif isinstance(out, dict) and 'logits' in out:
                    logits = out['logits']
                else:
                    logits = out
                
                # Fix: Use consistent argument order - check if loss_fn expects (output, target)
                try:
                    loss = loss_fn(logits, y)
                except TypeError:
                    # Try reversed order if first attempt fails
                    loss = loss_fn(y, logits)
            forward_time = time.time() - forward_start
            timings['forward_pass'] = forward_time
            
            # For compatibility, return logits as output
            out = logits
        else:
            # Standard tensor inputs (CNN models)
            if y is None:
                raise ValueError("Target tensor y must be provided for tensor inputs")
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            forward_start = time.time()
            with autocast('cuda'):
                out = self.model(x)
                # Fix: Use consistent argument order
                try:
                    loss = loss_fn(out, y)
                except TypeError:
                    # Try reversed order if first attempt fails
                    loss = loss_fn(y, out)
            forward_time = time.time() - forward_start
            timings['forward_pass'] = forward_time
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            self.anomaly_count += 1
            logger.warning(f"NaN/Inf loss detected (count: {self.anomaly_count})")
            
            if self.anomaly_count > self.max_anomalies:
                raise RuntimeError(f"Too many anomalies ({self.anomaly_count}), stopping training")
            
            # Skip this batch
            self.optimizer.zero_grad()
            
            # Return previous loss value if available
            if self.loss_history:
                return self.loss_history[-1], out.detach()
            else:
                return 0.0, out.detach()
        
        # Backward pass
        backward_start = time.time()
        
        # Enable anomaly detection if requested
        with torch.autograd.set_detect_anomaly(self.detect_anomaly):
            self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.grad_clip_norm is not None or self.grad_clip_value is not None:
            self.scaler.unscale_(self.optimizer)
            
            # Clip by norm
            if self.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.grad_clip_norm
                )
                
                # Log if clipping occurred
                if grad_norm > self.grad_clip_norm:
                    logger.debug(f"Gradient norm clipped: {grad_norm:.4f} -> {self.grad_clip_norm}")
            
            # Clip by value
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.grad_clip_value
                )
        
        backward_time = time.time() - backward_start
        timings['backward_pass'] = backward_time
        
        # Store and analyze gradients periodically
        grad_start = time.time()
        if self.steps % self.store_interval == 0:
            # Clean up removed parameters
            self._cleanup_removed_parameters()
            
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
            self._print_progress(loss_value, timings)
        
        # Update topology information periodically
        if self.steps % self.update_interval == 0:
            self._update_topology_async()
        
        # Periodic cleanup
        if self.steps % 100 == 0:
            self._cleanup_memory()
        
        return loss_value, out

    def step_complex(self, loss_fn: Callable, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                    y: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """
        Complex step method for handling different input formats.
        This is an alias to the main step method for backward compatibility.
        
        Args:
            loss_fn: Loss function
            batch: Input batch (tensor or dict)
            y: Target tensor (optional for dict inputs)
            
        Returns:
            Tuple of (loss_value, model_output)
        """
        return self.step(loss_fn, batch, y)

    def _cleanup_removed_parameters(self) -> None:
        """Remove parameter data for parameters that no longer exist in the model."""
        current_param_names = set(name for name, _ in self.model.named_parameters())
        removed_params = set(self.param_data.keys()) - current_param_names
        
        for param_name in removed_params:
            logger.debug(f"Removing data for parameter: {param_name}")
            del self.param_data[param_name]
            
            # Also remove from visualization data
            if param_name in self._visualization_data['gradient_stats']:
                del self._visualization_data['gradient_stats'][param_name]

    def _print_progress(self, loss_value: float, timings: dict) -> None:
        """Print training progress information."""
        total_time = sum(timings.values())
        print(f"Step {self.steps}: Loss = {loss_value:.6f}, Total time = {total_time:.3f}s")
        
        # Optional: Print detailed timings
        if self.steps % 100 == 0:
            for phase, time_val in timings.items():
                print(f"  {phase}: {time_val:.3f}s ({time_val/total_time*100:.1f}%)")

    def _cleanup_memory(self) -> None:
        """Periodic memory cleanup to prevent memory leaks."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_visualization_data(self):
        """Get visualization data for external analysis."""
        return {
            'loss_values': list(self._visualization_data['loss_values']),
            'valley_detections': list(self._visualization_data['valley_detections']),
            'bifurcations': list(self.bifurcations),
            'gradient_stats': {
                name: list(stats) for name, stats in self._visualization_data['gradient_stats'].items()
            }
        }

    @property
    def param_groups(self):
        """Access to underlying optimizer's parameter groups."""
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients of the base optimizer for PyTorch compatibility."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def _update_topology_async(self) -> None:
        """Update topology asynchronously with proper thread safety."""
        with self._state_lock:
            # Check if update is already in progress
            if self._update_in_progress:
                return

            # Check if previous future is done
            if hasattr(self, '_topology_update_future') and self._topology_update_future:
                if not self._topology_update_future.done():
                    return  # Previous update still running

                # Get result to check for exceptions
                try:
                    self._topology_update_future.result(timeout=0)
                except Exception as e:
                    logger.error(f"Previous topology update failed: {e}")

            # Mark update as in progress
            self._update_in_progress = True

            # Mark all parameters as pending update
            self._pending_updates = set(self.param_data.keys())

        # Submit update task
        def update_with_cleanup():
            try:
                self._update_topology()
            except Exception as e:
                logger.error(f"Topology update failed: {e}")
            finally:
                with self._state_lock:
                    self._pending_updates.clear()
                    self._update_in_progress = False

        try:
            self._topology_update_future = self.executor.submit(update_with_cleanup)
        except Exception as e:
            logger.error(f"Failed to submit topology update: {e}")
            # Reset state if submission fails
            with self._state_lock:
                self._pending_updates.clear()
                self._update_in_progress = False

    def state_dict(self):
        """Return optimizer state dict for checkpoint saving."""
        # Return a dict containing both base optimizer state and TALT-specific state
        state_dict = {
            'base_optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            # We're not saving internal tracking variables as they're regenerated
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict for checkpoint loading."""
        self.optimizer.load_state_dict(state_dict['base_optimizer'])
        self.steps = state_dict['steps']

    def shutdown(self) -> None:
        """Clean up resources."""
        # Cancel any pending tasks
        if hasattr(self, '_topology_update_future') and self._topology_update_future:
            try:
                self._topology_update_future.cancel()
            except Exception:
                pass  # Ignore cancellation errors

        # Shut down executor
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

        # Clear memory
        self.param_data.clear()
        self._visualization_data.clear()

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
