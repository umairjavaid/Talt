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
        
        # Test base_optimizer call to ensure it works and store the optimizer
        try:
            self.optimizer = base_optimizer(model.parameters(), lr=lr)
            if not hasattr(self.optimizer, 'step') or not hasattr(self.optimizer, 'zero_grad'):
                raise TypeError("base_optimizer must return a valid PyTorch optimizer")
        except Exception as e:
            raise ValueError(f"base_optimizer failed to create valid optimizer: {e}")
        
        # Store all parameters
        self.lr = lr
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
        self._state_lock = threading.RLock()  # Use RLock for reentrant locking
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
        self.grad_buffer = {}  # Temporary buffer for gradients

        # Register hooks for parameters of sufficient size
        registered_params = 0
        total_params = 0
        
        # Make threshold adaptive based on model size for better coverage
        total_model_params = sum(p.numel() for p in model.parameters())
        adaptive_threshold = max(10, min(self.min_param_size, int(total_model_params * 0.001)))
        
        logger.info(f"TALT initialized with projection_dim={projection_dim}, memory_size={memory_size}, update_interval={update_interval}")
        logger.info(f"Model has {total_model_params} total parameters")
        logger.info(f"Using adaptive threshold: {adaptive_threshold} (original: {self.min_param_size})")
        
        # Store registered hooks to prevent garbage collection
        self._hooks = []
        
        for name, p in model.named_parameters():
            total_params += 1
            if p.requires_grad and p.numel() > adaptive_threshold:
                registered_params += 1
                
                # Create dimension-reduced representation
                dim = p.numel()
                target_dim = max(1, min(self.projection_dim, dim // 4))  # Ensure at least 1 dimension

                self.param_data[name] = {
                    'projector': RandomProjection(dim, target_dim),
                    'covariance': IncrementalCovariance(target_dim, decay=cov_decay, device=self.device),
                    'valley_detector': ValleyDetector(window_size=5, threshold=0.2),
                    'valley_dirs': None,
                    'transformation': None,
                    'gradient_norm_history': deque(maxlen=50),
                    'consistency_history': deque(maxlen=50),
                    'grad_count': 0,
                    'last_grad': None,
                    'param_dim': dim,  # Store original dimension
                    'proj_dim': target_dim  # Store projected dimension
                }

                self._visualization_data['gradient_stats'][name] = deque(maxlen=max_visualization_points)
                self.grad_buffer[name] = deque(maxlen=memory_size)

                logger.debug(f"Registered TALT hook for {name} (size: {p.numel()}, proj_dim: {target_dim})")
                
                # Register gradient hook with proper closure
                def make_hook(param_name):
                    return lambda grad: self._transform_gradient(grad, param_name)
                
                hook = p.register_hook(make_hook(name))
                self._hooks.append(hook)
        
        logger.info(f"TALT: Registered hooks for {registered_params}/{total_params} parameters")
        if registered_params == 0:
            logger.warning("No parameters registered for TALT! Consider lowering min_param_size.")

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

        # Handle None gradients
        if grad is None:
            return grad

        # Make a copy of original gradient
        orig_grad = grad.clone()

        # Get parameter data with thread safety
        with self._state_lock:
            param_info = self.param_data.get(name)
            if param_info is None:
                return grad

        try:
            # Always store gradient information for covariance updates
            flat_grad = grad.view(-1)
            
            # Validate gradient tensor
            if torch.isnan(flat_grad).any() or torch.isinf(flat_grad).any():
                logger.warning(f"NaN/Inf gradient detected for {name}, skipping transformation")
                return orig_grad
            
            # Store gradient norm
            grad_norm = torch.norm(flat_grad).item()
            param_info['gradient_norm_history'].append((self.steps, grad_norm)) # Store step with norm
            param_info['grad_count'] += 1
            
            # Store gradient for later covariance update - ensure device consistency
            cpu_grad = flat_grad.detach().cpu()
            self.grad_buffer[name].append(cpu_grad)
            
            # Update valley detector with CPU tensor
            param_info['valley_detector'].update(cpu_grad)
            
            # Log periodically
            if self.steps % 100 == 0:
                logger.debug(f"Gradient hook for {name}: norm={grad_norm:.4f}, count={param_info['grad_count']}")
            
            # Check if transformation is available
            transformation = param_info.get('transformation')
            if transformation is None:
                return grad
            
            # PROJECT -> TRANSFORM -> UNPROJECT workflow
            # 1. Project gradient to low-dimensional space
            projected_grad = param_info['projector'].project(flat_grad)
            
            # Ensure transformation is on correct device
            if transformation.device != projected_grad.device:
                transformation = transformation.to(projected_grad.device)

            # 2. Check for valleys in the projected space
            is_valley, valley_dir = param_info['valley_detector'].detect_valley()

            if is_valley:
                logger.info(f"Valley detected for {name} at step {self.steps}")
                
                # Record bifurcation with thread safety
                with self._state_lock:
                    self.bifurcations.append(self.steps)
                    self._visualization_data['valley_detections'].append(
                        (self.steps, name, 'valley')
                    )

                # Apply valley transformation in projected space if valley_dir exists
                if valley_dir is not None:
                    # Valley direction is from the original space (where ValleyDetector operates).
                    # Move it to the correct device (same as projected_grad).
                    valley_dir = valley_dir.to(projected_grad.device)
                    
                    # Project valley_dir to the low-dimensional space.
                    projected_valley_dir = param_info['projector'].project(valley_dir)
                    
                    # Ensure no NaN/Inf in valley direction
                    if torch.isnan(projected_valley_dir).any() or torch.isinf(projected_valley_dir).any():
                        logger.warning(f"NaN/Inf in valley direction for {name}, skipping valley transformation")
                    else:
                        # Now projected_valley_dir is in the same (projected) space as projected_grad.
                        dot_product = torch.dot(projected_grad, projected_valley_dir)
                        valley_component = dot_product * projected_valley_dir
                        
                        # Clip valley component to prevent gradient explosion
                        valley_component = torch.clamp(valley_component, -10.0, 10.0)
                        
                        projected_grad = projected_grad + self.valley_strength * valley_component

            # 3. Apply curvature-based transformation in projected space
            transformed_projected = torch.matmul(transformation, projected_grad.unsqueeze(1)).squeeze(1)

            # 4. Unproject back to original space
            # Since we used random projection, we need to approximate the inverse
            # Using the transpose of the projection matrix scaled appropriately
            projection_matrix = param_info['projector'].projection
            if projection_matrix.is_sparse:
                projection_matrix = projection_matrix.to_dense()
            
            # Move to correct device
            projection_matrix = projection_matrix.to(transformed_projected.device)
            
            # Approximate inverse projection (transpose scaled by dimension ratio)
            scale_factor = param_info['param_dim'] / param_info['proj_dim']
            transformed_grad = scale_factor * torch.matmul(projection_matrix.t(), transformed_projected.unsqueeze(1)).squeeze(1)

            # Safety checks
            transformed_grad_flat = transformed_grad.view(-1)
            
            # Check for NaN/Inf in transformed gradient
            if torch.isnan(transformed_grad_flat).any() or torch.isinf(transformed_grad_flat).any():
                logger.warning(f"NaN/Inf in transformed gradient for {name}, using original")
                return orig_grad
            
            # Cosine similarity check
            orig_grad_flat = orig_grad.view(-1)
            if orig_grad_flat.numel() == transformed_grad_flat.numel():
                cos_sim = nn.functional.cosine_similarity(
                    transformed_grad_flat, orig_grad_flat, dim=0
                )

                if cos_sim < 0.6:
                    blend_factor = 0.6 - cos_sim
                    transformed_grad = (1.0 - blend_factor) * transformed_grad.view_as(orig_grad) + blend_factor * orig_grad
                else:
                    transformed_grad = transformed_grad.view_as(orig_grad)
            else:
                logger.warning(f"Dimension mismatch for {name}: {transformed_grad_flat.numel()} vs {orig_grad_flat.numel()}")
                return orig_grad

            return transformed_grad

        except Exception as e:
            logger.warning(f"Error transforming gradient for {name}: {e}")
            return orig_grad

    def _update_topology(self) -> None:
        """Update topology information for all tracked parameters."""
        logger.info(f"TALT topology update triggered at step {self.steps}")
        
        updated_params = 0
        for name, param_info in self.param_data.items():
            # Update covariance with buffered gradients
            if name in self.grad_buffer and len(self.grad_buffer[name]) >= 2:
                # Process all buffered gradients
                for grad in list(self.grad_buffer[name]):  # Create copy to avoid concurrent modification
                    try:
                        # Project gradient
                        projected_grad = param_info['projector'].project(grad.to(self.device))
                        # Update covariance
                        param_info['covariance'].update(projected_grad)
                    except Exception as e:
                        logger.warning(f"Error processing gradient for {name}: {e}")
                        continue
                
                # Clear buffer after processing
                self.grad_buffer[name].clear()
            
            # Check if we have enough covariance updates
            if param_info['covariance'].count < 3:
                logger.debug(f"  {name}: Insufficient covariance updates ({param_info['covariance'].count} < 3)")
                continue

            try:
                # Get the covariance matrix
                cov = param_info['covariance'].get_covariance()
                
                # Validate covariance matrix
                if torch.isnan(cov).any() or torch.isinf(cov).any():
                    logger.warning(f"NaN/Inf in covariance matrix for {name}, skipping")
                    continue

                # Adaptive regularization based on condition number
                if self.adaptive_reg:
                    try:
                        eigenvals = torch.linalg.eigvalsh(cov)
                        # Filter out negative eigenvalues due to numerical errors
                        eigenvals = eigenvals[eigenvals > 1e-12]
                        if len(eigenvals) > 0:
                            condition_number = eigenvals[-1] / eigenvals[0]
                            reg = max(1e-8, min(1e-2, 1e-5 * condition_number))
                        else:
                            reg = 1e-6
                        cov = cov + reg * torch.eye(cov.shape[0], device=cov.device)
                        logger.debug(f"  {name}: Applied adaptive regularization {reg:.2e}")
                    except RuntimeError as e:
                        # Fallback: add standard regularization
                        cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
                        logger.debug(f"  {name}: Applied fallback regularization due to: {e}")

                # Use power iteration for more stable eigendecomposition
                power_iter = PowerIteration(max_iter=20, tol=1e-5)
                eigenvalues, eigenvectors = power_iter.compute_eigenpairs(cov, k=min(5, cov.shape[0]))

                # Validate eigendecomposition results
                if torch.isnan(eigenvalues).any() or torch.isnan(eigenvectors).any():
                    logger.warning(f"NaN in eigendecomposition for {name}, skipping")
                    continue

                logger.debug(f"  {name}: Computed {len(eigenvalues)} eigenvalues: {eigenvalues[:3].tolist()}")

                # Create transformation matrix for gradient adjustment IN PROJECTED SPACE
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

                    # Clamp scale to prevent extreme transformations
                    scale = max(0.1, min(scale, 5.0))
                    transform += (scale - 1.0) * torch.matmul(vec, vec.t())

                # Store transformation matrix (for projected space)
                # transform is already on self.device because cov is on self.device
                param_info['transformation'] = transform

                # Store visualization data with thread safety
                grad_norm_avg = (
                    np.mean([item[1] for item in param_info['gradient_norm_history']])
                    if param_info['gradient_norm_history']
                    else 0.0
                )
                with self._state_lock:
                    self._visualization_data['gradient_stats'][name].append({
                        'step': self.steps,
                        'eigenvalues': eigenvalues[:min(3, len(eigenvalues))].detach().cpu().numpy(),
                        'grad_norm': grad_norm_avg
                    })

                updated_params += 1
                logger.info(f"  {name}: Successfully updated transformation matrix")

                # Clean up to avoid memory leaks
                del cov, eigenvalues, eigenvectors, transform

            except Exception as e:
                logger.warning(f"Error updating topology for {name}: {e}")
        
        logger.info(f"TALT topology update completed: {updated_params}/{len(self.param_data)} parameters updated")

    def step(self, loss_fn: Callable, x: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple, List], 
             y: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """
        Perform an optimization step with gradient clipping and NaN detection.
        
        Args:
            loss_fn: Loss function that takes (predictions, targets) and returns loss
            x: Input data - tensor (CNN), dict (BERT/transformers), or tuple/list of (input, target)
            y: Target data - if None, extracted from x['labels'] for dict inputs or x[1] for tuple inputs
            
        Returns:
            Tuple of (loss_value, model_output)
        """
        # Validate batch structure and handle tuple/list inputs
        if isinstance(x, (tuple, list)):
            # Handle tuple/list format: (input, target)
            if len(x) >= 2:
                input_data, y = x[0], x[1]
                x = input_data
            else:
                raise ValueError(f"Tuple/list input must have at least 2 elements, got {len(x)}")
        elif not isinstance(x, (torch.Tensor, dict)):
            raise TypeError(f"Unsupported batch type: {type(x)}. Expected tensor, dict, tuple, or list.")
        
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
                
                # Use consistent argument order
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
                # Use consistent argument order
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

    def diagnose_visualization_state(self):
        """Print diagnostic information about TALT state for debugging."""
        print("\n=== TALT Diagnostic Report ===")
        print(f"Steps completed: {self.steps}")
        print(f"Parameters tracked: {len(self.param_data)}")
        print(f"Update interval: {self.update_interval}, Store interval: {self.store_interval}")
        
        for name, info in self.param_data.items():
            print(f"\n{name}:")
            print(f"  Gradient norms collected: {len(info['gradient_norm_history'])}")
            print(f"  Covariance updates: {info['covariance'].count}")
            print(f"  Has transformation: {info.get('transformation') is not None}")
            print(f"  Grad count: {info['grad_count']}")
            print(f"  Buffered gradients: {len(self.grad_buffer.get(name, []))}")
            print(f"  Original dim: {info['param_dim']}, Projected dim: {info['proj_dim']}")
            valley_detections = [d for d in self._visualization_data['valley_detections'] if d[1] == name]
            print(f"  Valley detections: {len(valley_detections)}")
        
        print(f"\nTotal bifurcations: {len(self.bifurcations)}")
        print(f"Total valley detections: {len(self._visualization_data['valley_detections'])}")
        print(f"Visualization data keys: {list(self._visualization_data.keys())}")
        
        # Check visualization stats
        total_viz_stats = sum(len(stats) for stats in self._visualization_data['gradient_stats'].values())
        print(f"Total visualization stats collected: {total_viz_stats}")
        print("===========================\n")

    def force_topology_update(self):
        """Force immediate topology update for testing/debugging."""
        logger.info("Forcing TALT topology update...")
        with self._state_lock:
            # Clear pending updates to allow forced update
            self._pending_updates.clear()
            self._update_in_progress = False
        self._update_topology()

    def get_visualization_data(self):
        """Get visualization data for external analysis."""
        logger.info("Collecting TALT visualization data...")
        
        # Create thread-safe copy of visualization data
        with self._state_lock:
            viz_data = {
                'loss_values': list(self._visualization_data['loss_values']),
                'loss_history': list(self.loss_history),
                'valley_detections': list(self._visualization_data['valley_detections']),
                'bifurcations': list(self.bifurcations),
                'gradient_stats': {},
                'eigenvalues_history': {},
                'gradient_norms_history': {}
            }
            
            # Deep copy gradient stats
            for name, stats_deque in self._visualization_data['gradient_stats'].items():
                viz_data['gradient_stats'][name] = list(stats_deque)
        
        # Process gradient stats to extract eigenvalues and norms
        for name, stats_deque in self._visualization_data['gradient_stats'].items():
            if len(stats_deque) > 0:
                stats_list = list(stats_deque)
                viz_data['gradient_stats'][name] = stats_list
                
                # Extract eigenvalues and gradient norms for easier access
                eigenvals = []
                grad_norms = []
                steps = []
                
                for stat_entry in stats_list:
                    if isinstance(stat_entry, dict):
                        if 'eigenvalues' in stat_entry:
                            eigenvals.append(stat_entry['eigenvalues'])
                        if 'grad_norm' in stat_entry:
                            grad_norms.append(stat_entry['grad_norm'])
                        if 'step' in stat_entry:
                            steps.append(stat_entry['step'])
                
                if eigenvals:
                    viz_data['eigenvalues_history'][name] = {
                        'eigenvalues': eigenvals,
                        'steps': steps
                    }
                
                if grad_norms:
                    viz_data['gradient_norms_history'][name] = {
                        'grad_norms': grad_norms,
                        'steps': steps
                    }
        
        # Add gradient norm history from param_data
        for param_name, param_info in self.param_data.items():
            if 'gradient_norm_history' in param_info and len(param_info['gradient_norm_history']) > 0:
                if param_name not in viz_data['gradient_norms_history']:
                    viz_data['gradient_norms_history'][param_name] = {}
                
                # Merge with existing data or create new
                existing_norms_list = viz_data['gradient_norms_history'][param_name].get('grad_norms', [])
                
                param_history_tuples = list(param_info['gradient_norm_history'])
                current_param_steps = [item[0] for item in param_history_tuples]
                current_param_norms = [item[1] for item in param_history_tuples]
                
                # Use the data from param_info if it's more extensive (longer)
                if len(current_param_norms) > len(existing_norms_list):
                    viz_data['gradient_norms_history'][param_name]['grad_norms'] = current_param_norms
                    viz_data['gradient_norms_history'][param_name]['steps'] = current_param_steps
        
        # Log summary
        logger.info(f"TALT visualization data collected: "
                    f"{len(viz_data['gradient_stats'])} params with stats, "
                    f"{len(viz_data['eigenvalues_history'])} params with eigenvalues, "
                    f"{len(viz_data['gradient_norms_history'])} params with gradient norms, "
                    f"{len(viz_data['valley_detections'])} valley detections, "
                    f"{len(viz_data['bifurcations'])} bifurcations")
        
        return viz_data

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
            if self._topology_update_future is not None:
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
        if self._topology_update_future is not None:
            try:
                self._topology_update_future.cancel()
            except Exception:
                pass  # Ignore cancellation errors

        # Shut down executor
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True) # Removed invalid timeout parameter

        # Remove gradient hooks to prevent memory leaks
        for hook in getattr(self, '_hooks', []):
            try:
                hook.remove()
            except Exception:
                pass  # Hook may already be removed

        # Clear memory
        self.param_data.clear()
        self.grad_buffer.clear()
        self._visualization_data.clear()

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup

    def get_tensorboard_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for TensorBoard logging.
        
        Returns:
            Dictionary containing current TALT metrics for TensorBoard
        """
        metrics = {}
        
        # Extract current eigenvalues
        eigenvalue_data = {}
        gradient_norms = {}
        curvature_estimates = {}
        gradient_transformations = {}
        
        for param_name, param_info in self.param_data.items():
            # Current gradient norms
            if 'gradient_norm_history' in param_info and param_info['gradient_norm_history']:
                gradient_norms[param_name] = param_info['gradient_norm_history'][-1][1] # Access norm from (step, norm)
            
            # Current eigenvalues from recent gradient stats
            if hasattr(self, '_visualization_data'):
                viz_data = self._visualization_data
                if 'gradient_stats' in viz_data and param_name in viz_data['gradient_stats']:
                    stats = list(viz_data['gradient_stats'][param_name])
                    if stats and isinstance(stats[-1], dict):
                        stat_entry = stats[-1]
                        if 'eigenvalues' in stat_entry:
                            eigenvalue_data[param_name] = stat_entry['eigenvalues']
                        
                        # Calculate curvature estimate from eigenvalues
                        if 'eigenvalues' in stat_entry:
                            eigenvals = np.array(stat_entry['eigenvalues'])
                            if len(eigenvals) > 0:
                                curvature_estimates[param_name] = float(np.max(eigenvals))
            
            # Gradient transformation metrics
            if param_info.get('transformation') is not None:
                # Calculate transformation effect (simplified)
                gradient_transformations[param_name] = {
                    'has_transformation': 1.0,
                    'transformation_norm': float(torch.norm(param_info['transformation']).item())
                }
        
        # Add metrics to result
        if eigenvalue_data:
            metrics['eigenvalues'] = eigenvalue_data
        if gradient_norms:
            metrics['gradient_norms'] = gradient_norms
        if curvature_estimates:
            metrics['curvature_estimates'] = curvature_estimates
        if gradient_transformations:
            metrics['gradient_transformations'] = gradient_transformations
        
        # Valley detections from recent visualization data
        recent_valley_detections = []
        if hasattr(self, '_visualization_data') and 'valley_detections' in self._visualization_data:
            # Get detections from last few steps
            for detection in list(self._visualization_data['valley_detections'])[-5:]:
                if isinstance(detection, (tuple, list)) and len(detection) >= 2:
                    recent_valley_detections.append(detection)
        
        if recent_valley_detections:
            metrics['valley_detections'] = recent_valley_detections
        
        # Bifurcations
        if hasattr(self, 'bifurcations') and self.bifurcations:
            metrics['bifurcations'] = list(self.bifurcations)
        
        return metrics