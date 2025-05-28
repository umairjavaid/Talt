"""Speed-Optimized TALT Optimizer - Simple, fast, and error-free implementation."""
from collections import deque
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import torch
import torch.nn as nn
import logging
import numpy as np
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ImprovedTALTOptimizer:
    """
    Speed-Optimized TALT Optimizer with maximum stability.
    
    Key changes for stability:
    - No torch.compile (causes errors)
    - Simplified gradient transformation
    - Robust error handling
    - Minimal branching in hot paths
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: Callable,
        *,
        lr: float = 1e-2,
        memory_size: int = 10,
        update_interval: int = 20,
        valley_strength: float = 0.05,  # Reduced from 0.1
        smoothing_factor: float = 0.5,  # Increased from 0.3
        grad_store_interval: int = 5,
        min_param_size: int = 100,
        max_param_size: int = 100000,  # Reduced from 1000000 to prevent memory issues
        device: Union[str, torch.device] = "cuda",
        gradient_clip_norm: float = 5.0,  # Reduced from 10.0
        min_eigenvalue: float = 1e-4,  # Increased from 1e-5 for stability
        regularization_strength: float = 1e-2,  # Increased from 1e-3
        # New theoretical fix parameters
        use_adaptive_memory: bool = True,
        use_gradient_smoothing: bool = True,
        smoothing_beta: float = 0.95,  # Increased from 0.9
        use_adaptive_thresholds: bool = True,
        use_parameter_normalization: bool = True,
        use_incremental_covariance: bool = True,
        eigenspace_blend_factor: float = 0.9,  # Increased from 0.7
        min_memory_ratio: float = 0.05,  # Reduced from 0.1
        # New stability parameters
        max_gradient_scale: float = 1.5,  # Reduced from 2.0
        min_gradient_scale: float = 0.2,  # Increased from 0.1
        stability_threshold: float = 1e-3,  # Threshold for stable gradients
        emergency_fallback: bool = True,  # Use emergency fallback on instability
        # Memory management parameters
        max_covariance_size: int = 10000,  # Maximum size for covariance matrix
        force_cpu_eigendecomp: bool = True  # Force CPU eigendecomposition for large matrices
    ):
        """Initialize speed-optimized TALT optimizer with enhanced stability."""
        self.model = model
        self.optimizer = base_optimizer(model.parameters(), lr=lr)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Initialize GradScaler only for CUDA
        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # TALT parameters - more conservative
        self.memory_size = memory_size
        self.update_interval = update_interval
        self.valley_strength = valley_strength
        self.smoothing_factor = smoothing_factor
        self.store_interval = grad_store_interval
        self.min_param_size = min_param_size
        self.max_param_size = max_param_size
        
        # Enhanced stability parameters
        self.gradient_clip_norm = gradient_clip_norm
        self.min_eigenvalue = min_eigenvalue
        self.regularization_strength = regularization_strength
        self.max_gradient_scale = max_gradient_scale
        self.min_gradient_scale = min_gradient_scale
        self.stability_threshold = stability_threshold
        self.emergency_fallback = emergency_fallback
        
        # Memory management parameters
        self.max_covariance_size = max_covariance_size
        self.force_cpu_eigendecomp = force_cpu_eigendecomp
        
        # Theoretical fixes parameters
        self.use_adaptive_memory = use_adaptive_memory
        self.use_gradient_smoothing = use_gradient_smoothing
        self.smoothing_beta = smoothing_beta
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.use_parameter_normalization = use_parameter_normalization
        self.use_incremental_covariance = use_incremental_covariance
        self.eigenspace_blend_factor = eigenspace_blend_factor
        self.min_memory_ratio = min_memory_ratio
        
        # State tracking
        self.steps = 0
        self.loss_history = []
        self.bifurcations = []
        
        # Per-parameter state
        self.param_data = {}
        self._hook_handles = {}
        
        # Initialize grad_change_rate tracking
        self.grad_change_rate = {}
        
        # Stability tracking
        self.instability_count = {}
        self.emergency_mode = False
        self.stable_steps = 0
        
        # Pre-compute constants
        self.valley_scale = 1.0 + self.valley_strength
        self.valley_threshold = 0.1  # Increased from 0.05
        
        # Visualization data
        self._visualization_data = {
            'loss_values': [],
            'bifurcations': [],
            'eigenvalues': {}
        }
        
        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameter tracking with adaptive memory sizes and fixes."""
        registered_params = 0
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            param_size = param.numel()
            
            if self.min_param_size <= param_size <= self.max_param_size:
                registered_params += 1
                
                # FIX 1: Adaptive memory size based on parameter dimension
                if self.use_adaptive_memory:
                    adaptive_memory_size = max(
                        self.memory_size,
                        min(int(np.sqrt(param_size) * 2), 50),  # 2*sqrt(d), max 50
                        int(param_size * self.min_memory_ratio)  # 10% of dimension
                    )
                else:
                    adaptive_memory_size = self.memory_size
                
                # Enhanced data structure with fixes
                self.param_data[name] = {
                    'grad_memory': deque(maxlen=adaptive_memory_size),
                    'principal_dirs': None,
                    'eigenvalues': None,
                    'size': param_size,
                    'adaptive_memory_size': adaptive_memory_size,
                    'param_dim': param_size,
                    
                    # FIX 4: Gradient smoothing
                    'smoothed_grad': None if self.use_gradient_smoothing else None,
                    
                    # FIX 6: Parameter-specific normalization
                    'grad_stats': {
                        'mean': 0.0,
                        'var': 1.0,
                        'count': 0
                    } if self.use_parameter_normalization else None,
                    
                    # FIX 5: Incremental covariance
                    'incremental_cov': None if self.use_incremental_covariance else None,
                    'n_samples': 0,
                    
                    # FIX 8: Transformation stability
                    'prev_transform': None,
                    
                    # FIX 2: Dynamic updates
                    'last_grad': None,
                    
                    # FIX 7: Progress tracking
                    'progress_tracker': {
                        'loss_history': deque(maxlen=20),
                        'grad_norm_history': deque(maxlen=20)
                    },
                    
                    # THEORETICAL FIX 4: Parameter-specific normalization (backward compatibility)
                    'grad_norm_ema': 1.0,
                    'grad_norm_alpha': 0.9,
                    # THEORETICAL FIX 3: Exponential moving average covariance
                    'cov_ema': None,
                    'cov_alpha': 0.95,
                    'eigenvalue_history': deque(maxlen=10)
                }
                
                # Initialize change rate tracking
                self.grad_change_rate[name] = 0.0
                
                # Register gradient hook
                handle = param.register_hook(
                    lambda grad, name=name: self._transform_gradient(grad, name)
                )
                self._hook_handles[name] = handle
                
                # Initialize visualization data
                self._visualization_data['eigenvalues'][name] = []
        
        logger.info(f"Enhanced TALT: Tracking {registered_params} parameters with theoretical fixes enabled")

    def _transform_gradient(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """Transform gradient with enhanced numerical stability."""
        if grad is None or name not in self.param_data:
            return grad
            
        # Early NaN/Inf detection with immediate fallback
        if not torch.isfinite(grad).all():
            logger.warning(f"Non-finite gradient detected for {name}")
            self.instability_count[name] = self.instability_count.get(name, 0) + 1
            
            # Emergency fallback: return zero gradient to prevent cascade
            if self.emergency_fallback and self.instability_count[name] > 3:
                self.emergency_mode = True
                return torch.zeros_like(grad)
            else:
                # Try to recover with clipped gradient
                grad_clipped = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                grad_norm = grad_clipped.norm()
                if grad_norm > self.gradient_clip_norm:
                    grad_clipped = grad_clipped * (self.gradient_clip_norm / grad_norm)
                return grad_clipped
        
        param_ref = self.param_data[name]
        
        # Reset instability counter on good gradient
        self.instability_count[name] = max(0, self.instability_count.get(name, 0) - 1)
        
        # Check gradient stability
        grad_norm = grad.norm().item()
        if grad_norm < self.stability_threshold:
            # Very small gradient - minimal transformation
            return grad * 0.95  # Slight damping
        
        # If in emergency mode, use conservative transformations only
        if self.emergency_mode:
            if self.stable_steps > 50:  # Reset after stability
                self.emergency_mode = False
                self.stable_steps = 0
            else:
                self.stable_steps += 1
                # Conservative gradient clipping only
                if grad_norm > self.gradient_clip_norm:
                    return grad * (self.gradient_clip_norm / grad_norm)
                return grad
        
        # FIX 6: Parameter-specific gradient statistics and normalization
        grad_normalized = grad
        if self.use_parameter_normalization and param_ref['grad_stats'] is not None:
            stats = param_ref['grad_stats']
            
            # Update running statistics with bounds checking
            stats['count'] += 1
            delta = grad_norm - stats['mean']
            stats['mean'] += delta / stats['count']
            delta2 = grad_norm - stats['mean']
            stats['var'] = 0.95 * stats['var'] + 0.05 * delta * delta2  # More stable update
            stats['var'] = max(stats['var'], 1e-8)  # Prevent zero variance
            
            # Conservative normalization
            norm_scale = min(2.0, 1.0 / max(np.sqrt(stats['var']), 1e-4))
            grad_normalized = grad * norm_scale
        
        # FIX 2: Dynamic update based on gradient change rate
        if param_ref['last_grad'] is not None:
            change = torch.norm(grad - param_ref['last_grad']) / (torch.norm(grad) + 1e-8)
            self.grad_change_rate[name] = 0.95 * self.grad_change_rate[name] + 0.05 * change.item()
            
            # Update more frequently if gradients changing rapidly
            if self.grad_change_rate[name] > 0.3 and self.steps % 10 == 0:  # Less frequent updates
                self._update_single_param_topology(name)
        
        param_ref['last_grad'] = grad.clone()
        
        # FIX 7: Progress tracking for selective amplification
        tracker = param_ref['progress_tracker']
        tracker['grad_norm_history'].append(grad_norm)
        
        # Conservative gradient clipping on normalized gradient
        normalized_grad_norm = grad_normalized.norm()
        if normalized_grad_norm > self.gradient_clip_norm:
            grad_normalized = grad_normalized * (self.gradient_clip_norm / normalized_grad_norm)
        
        dirs = param_ref['principal_dirs']
        vals = param_ref['eigenvalues']
        
        if dirs is None or vals is None:
            return grad_normalized if self.use_parameter_normalization else grad
        
        try:
            flat_grad = grad_normalized.view(-1)
            
            # Ensure same device
            if dirs.device != grad_normalized.device:
                dirs = dirs.to(grad_normalized.device)
                param_ref['principal_dirs'] = dirs
            if vals.device != grad_normalized.device:
                vals = vals.to(grad_normalized.device)
                param_ref['eigenvalues'] = vals
            
            # Project gradient with bounds checking
            n_components = min(dirs.shape[1], vals.shape[0], param_ref['adaptive_memory_size'])
            dirs_subset = dirs[:, :n_components]
            vals_subset = vals[:n_components]
            
            # Check for numerical issues in eigenvectors
            if not torch.isfinite(dirs_subset).all() or not torch.isfinite(vals_subset).all():
                logger.warning(f"Non-finite eigendata for {name}, skipping transformation")
                return grad_normalized if self.use_parameter_normalization else grad
            
            # Compute coefficients with stability check
            try:
                coeffs = torch.mv(dirs_subset.t(), flat_grad)
                if not torch.isfinite(coeffs).all():
                    return grad_normalized if self.use_parameter_normalization else grad
            except Exception:
                return grad_normalized if self.use_parameter_normalization else grad
            
            # FIX 3: More conservative adaptive thresholds
            eigenval_abs = vals_subset.abs()
            if self.use_adaptive_thresholds:
                # Use safer percentiles
                valley_threshold = torch.quantile(eigenval_abs, 0.3)  # Bottom 30% (was 20%)
                high_curve_threshold = torch.quantile(eigenval_abs, 0.7)  # Top 30% (was 20%)
            else:
                valley_threshold = self.valley_threshold
                high_curve_threshold = 1.0
            
            # FIX 7: Much more conservative valley strength
            effective_valley_strength = self.valley_strength * 0.5  # Start with 50% of original
            if len(tracker['grad_norm_history']) > 10:
                recent_norms = list(tracker['grad_norm_history'])[-10:]
                norm_variance = np.var(recent_norms)
                mean_norm = np.mean(recent_norms)
                
                # Even more conservative in stationary regions
                if norm_variance < 0.01 * mean_norm**2:
                    effective_valley_strength = self.valley_strength * 0.1
            
            # Apply very conservative scaling
            scales = torch.ones_like(vals_subset)
            
            # Valley amplification with strict bounds
            valley_mask = eigenval_abs < valley_threshold
            valley_scale = min(1.0 + effective_valley_strength, self.max_gradient_scale)
            scales = torch.where(valley_mask, 
                               torch.tensor(valley_scale, device=scales.device), 
                               scales)
            
            # High curvature damping with bounds
            high_curve_mask = eigenval_abs > high_curve_threshold
            damping = self.smoothing_factor / torch.sqrt(eigenval_abs.clamp(min=self.min_eigenvalue))
            damping = torch.clamp(damping, min=self.min_gradient_scale, max=1.0)
            scales = torch.where(high_curve_mask, damping, scales)
            
            # Apply scaling with safety checks
            scaled_coeffs = coeffs * scales
            
            # Check for numerical explosion
            if not torch.isfinite(scaled_coeffs).all():
                logger.warning(f"Non-finite coefficients after scaling for {name}")
                return grad_normalized if self.use_parameter_normalization else grad
            
            # Project back with safety
            try:
                new_transform = torch.mv(dirs_subset, scaled_coeffs)
                if not torch.isfinite(new_transform).all():
                    return grad_normalized if self.use_parameter_normalization else grad
            except Exception:
                return grad_normalized if self.use_parameter_normalization else grad
            
            # FIX 8: More conservative eigenspace transitions
            if param_ref['prev_transform'] is not None:
                # Higher blend factor for stability
                transformed_grad = (
                    self.eigenspace_blend_factor * new_transform + 
                    (1 - self.eigenspace_blend_factor) * param_ref['prev_transform']
                )
            else:
                transformed_grad = new_transform
            
            param_ref['prev_transform'] = transformed_grad.clone()
            
            # Final safety check on transformed gradient
            transform_norm = transformed_grad.norm()
            original_norm = flat_grad.norm()
            
            # Prevent excessive scaling
            if transform_norm > original_norm * self.max_gradient_scale:
                scale_factor = (original_norm * self.max_gradient_scale) / transform_norm
                transformed_grad = transformed_grad * scale_factor
            elif transform_norm < original_norm * self.min_gradient_scale:
                scale_factor = (original_norm * self.min_gradient_scale) / transform_norm
                transformed_grad = transformed_grad * scale_factor
            
            # Record bifurcation if valleys detected (less frequently)
            if valley_mask.any() and self.steps % 5 == 0:  # Record less frequently
                self.bifurcations.append(self.steps)
                self._visualization_data['bifurcations'].append(self.steps)
            
            # Final gradient with denormalization
            if self.use_parameter_normalization and param_ref['grad_stats'] is not None:
                final_grad = transformed_grad.view_as(grad) * max(np.sqrt(param_ref['grad_stats']['var']), 1e-4)
            else:
                final_grad = transformed_grad.view_as(grad)
            
            # Ultimate safety check
            if not torch.isfinite(final_grad).all():
                logger.warning(f"Final gradient non-finite for {name}, using fallback")
                return grad_normalized if self.use_parameter_normalization else grad
            
            return final_grad
            
        except Exception as e:
            logger.warning(f"Enhanced gradient transformation failed for {name}: {e}")
            return grad_normalized if self.use_parameter_normalization else grad

    def _update_single_param_topology(self, name: str) -> None:
        """Update topology for a single parameter (for dynamic updates)."""
        if name not in self.param_data:
            return
            
        param_ref = self.param_data[name]
        if len(param_ref['grad_memory']) < 3:
            return
            
        try:
            self._update_parameter_eigenspace(name, param_ref)
        except Exception as e:
            logger.debug(f"Single parameter topology update failed for {name}: {e}")

    def _update_parameter_eigenspace(self, name: str, param_ref: Dict) -> None:
        """Update eigenspace for a specific parameter with enhanced numerical stability and memory management."""
        try:
            # More conservative memory requirements
            if len(param_ref['grad_memory']) < 5:  # Increased from 3
                return
            
            # Check parameter size and skip if too large to prevent memory issues
            param_size = param_ref['param_dim']
            if param_size > self.max_covariance_size:
                logger.debug(f"Skipping eigenspace update for {name} (size {param_size} > {self.max_covariance_size})")
                return
            
            # FIX 5: Use incremental PCA and enhanced covariance estimation
            if self.use_incremental_covariance:
                # Incremental covariance update with stability checks
                new_grads = list(param_ref['grad_memory'])[-2:]  # Even fewer gradients for stability
                for grad in new_grads:
                    # Force CPU computation for large gradients
                    if param_size > 5000 or self.force_cpu_eigendecomp:
                        grad_cpu = grad.cpu()
                    else:
                        grad_cpu = grad.to(self.device)
                    
                    # Check gradient validity
                    if not torch.isfinite(grad_cpu).all():
                        continue
                    
                    # Limit gradient size to prevent memory explosion
                    if grad_cpu.numel() > self.max_covariance_size:
                        logger.warning(f"Gradient too large for {name}, skipping covariance update")
                        continue
                    
                    if param_ref['incremental_cov'] is None:
                        param_ref['incremental_cov'] = torch.outer(grad_cpu, grad_cpu)
                        param_ref['n_samples'] = 1
                    else:
                        # More conservative exponentially weighted update
                        alpha = min(1.0 / (param_ref['n_samples'] + 1), 0.02)  # Reduced from 0.05
                        new_cov = torch.outer(grad_cpu, grad_cpu)
                        
                        # Check for numerical stability and memory
                        if torch.isfinite(new_cov).all() and new_cov.numel() < self.max_covariance_size * self.max_covariance_size:
                            # Move to CPU if needed
                            if param_ref['incremental_cov'].device != grad_cpu.device:
                                param_ref['incremental_cov'] = param_ref['incremental_cov'].to(grad_cpu.device)
                            
                            param_ref['incremental_cov'] = (
                                (1 - alpha) * param_ref['incremental_cov'] + 
                                alpha * new_cov
                            )
                            param_ref['n_samples'] += 1
                        else:
                            logger.warning(f"Skipping covariance update for {name} due to size or stability issues")
                
                cov = param_ref['incremental_cov']
                if cov is None:
                    return
                    
                # Force CPU computation for eigendecomposition if matrix is large
                if cov.numel() > 100000 or self.force_cpu_eigendecomp:  # 100k elements threshold
                    cov = cov.cpu()
            else:
                # Original method with more stability checks
                grad_list = list(param_ref['grad_memory'])
                
                # Filter out any non-finite gradients
                valid_grads = []
                for g in grad_list:
                    g_gpu = g.to(self.device)
                    if torch.isfinite(g_gpu).all():
                        valid_grads.append(g_gpu)
                
                if len(valid_grads) < 3:
                    return
                
                grad_tensor = torch.stack(valid_grads)
                
                if grad_tensor.std() < 1e-6:  # Increased threshold
                    return
                
                # THEORETICAL FIX 3: More stable exponential moving average for covariance
                recent_grad = grad_tensor[-1]
                
                if param_ref['cov_ema'] is None:
                    k = grad_tensor.shape[0]
                    if k > 1:
                        cov = torch.matmul(grad_tensor.t(), grad_tensor) / (k - 1)
                        param_ref['cov_ema'] = cov
                    else:
                        return
                else:
                    new_cov = torch.outer(recent_grad, recent_grad)
                    if torch.isfinite(new_cov).all():
                        param_ref['cov_ema'] = (
                            param_ref['cov_alpha'] * param_ref['cov_ema'] + 
                            (1 - param_ref['cov_alpha']) * new_cov
                        )
                
                cov = param_ref['cov_ema']
                if cov is None:
                    return
            
            # Enhanced regularization for stability
            reg_value = max(self.regularization_strength * cov.trace() / cov.shape[0], 1e-5)
            cov = cov + reg_value * torch.eye(cov.shape[0], device=cov.device)
            
            # Check covariance matrix validity and size
            if not torch.isfinite(cov).all() or cov.trace() <= 0:
                logger.warning(f"Invalid covariance matrix for {name}")
                return
            
            # FIXED: Proper covariance matrix size check
            # Check total number of elements in covariance matrix, not just dimensions
            cov_elements = cov.numel()
            max_allowed_elements = self.max_covariance_size * self.max_covariance_size
            
            if cov_elements > max_allowed_elements:
                logger.warning(f"Covariance matrix too large for {name}: {cov.shape} ({cov_elements} elements > {max_allowed_elements})")
                return
            
            # Also check if individual dimension is reasonable for eigendecomposition
            if cov.shape[0] > 2000:  # Conservative limit for eigendecomposition
                logger.debug(f"Skipping eigendecomposition for {name} due to large dimension: {cov.shape[0]}")
                return
            
            # Eigendecomposition with error handling and memory management
            try:
                # Force CPU computation for large matrices
                if cov.numel() > 50000 or self.force_cpu_eigendecomp:
                    cov_cpu = cov.cpu()
                    eigenvals, eigenvecs = torch.linalg.eigh(cov_cpu)
                else:
                    eigenvals, eigenvecs = torch.linalg.eigh(cov)
                
                # Filter out very small eigenvalues more aggressively
                valid_mask = eigenvals.abs() > self.min_eigenvalue * 10  # 10x threshold
                if not valid_mask.any():
                    return
                
                eigenvals = eigenvals[valid_mask]
                eigenvecs = eigenvecs[:, valid_mask]
                
                # Sort by magnitude and keep fewer components for stability
                idx = eigenvals.abs().argsort(descending=True)
                n_components = min(
                    max(3, int(np.sqrt(param_ref['param_dim']) * 0.5)) if self.use_adaptive_memory else param_ref['adaptive_memory_size'] // 2,
                    len(eigenvals) // 2,  # Use at most half the eigenvalues
                    eigenvals.shape[0]
                )
                idx = idx[:n_components]
                
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
                # Final validity check
                if not torch.isfinite(eigenvals).all() or not torch.isfinite(eigenvecs).all():
                    logger.warning(f"Non-finite eigendecomposition for {name}")
                    return
                
                # Store results
                param_ref['eigenvalues'] = eigenvals
                param_ref['principal_dirs'] = eigenvecs
                
                # THEORETICAL FIX 2: Update eigenvalue history for adaptive thresholds
                param_ref['eigenvalue_history'].append(eigenvals.clone())
                
                # Update visualization (less frequently)
                if len(eigenvals) >= 3:
                    top_vals = eigenvals[:3].detach().cpu().numpy()
                else:
                    top_vals = eigenvals.detach().cpu().numpy()
                self._visualization_data['eigenvalues'][name].append(top_vals)
                
            except (torch.linalg.LinAlgError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU OOM during eigendecomposition for {name}, clearing eigendata")
                    torch.cuda.empty_cache()
                else:
                    logger.warning(f"Eigendecomposition failed for {name}: {e}")
                # Clear eigendata to prevent using stale data
                param_ref['eigenvalues'] = None
                param_ref['principal_dirs'] = None
                return
                
        except Exception as e:
            logger.warning(f"Parameter eigenspace update failed for {name}: {e}")
            # Clear eigendata on any error
            param_ref['eigenvalues'] = None
            param_ref['principal_dirs'] = None

    def _store_gradients(self):
        """Store gradients with smoothing and normalization fixes."""
        for name, param_ref in self.param_data.items():
            # Get parameter by name
            param = None
            for n, p in self.model.named_parameters():
                if n == name:
                    param = p
                    break
            
            if param is None or param.grad is None:
                continue
            
            grad = param.grad.detach()
            
            # Check validity
            if not torch.isfinite(grad).all():
                continue
            
            # FIX 4: Exponential moving average of gradients (smoothing)
            if self.use_gradient_smoothing:
                if param_ref['smoothed_grad'] is None:
                    param_ref['smoothed_grad'] = grad.clone()
                else:
                    param_ref['smoothed_grad'] = (
                        param_ref['smoothed_grad'] * self.smoothing_beta + 
                        grad * (1 - self.smoothing_beta)
                    )
                
                # Store smoothed gradient instead of raw gradient
                grad_to_store = param_ref['smoothed_grad']
            else:
                grad_to_store = grad
            
            # FIX 6: Parameter-specific normalization before storing
            if self.use_parameter_normalization and param_ref['grad_stats'] is not None:
                stats = param_ref['grad_stats']
                grad_norm = grad_to_store.norm().item()
                if grad_norm > 0:
                    # Update running average
                    stats['count'] += 1
                    delta = grad_norm - stats['mean']
                    stats['mean'] += delta / stats['count']
                    stats['var'] = 0.9 * stats['var'] + 0.1 * delta * delta
                    
                    # Store normalized gradient
                    normalized_grad = grad_to_store / max(np.sqrt(stats['var']), 1e-8)
                    grad_flat = normalized_grad.view(-1).cpu()
                else:
                    grad_flat = grad_to_store.view(-1).cpu()
            else:
                # THEORETICAL FIX 4: Backward compatibility normalization
                grad_norm = grad_to_store.norm().item()
                if grad_norm > 0:
                    param_ref['grad_norm_ema'] = (
                        param_ref['grad_norm_alpha'] * param_ref['grad_norm_ema'] + 
                        (1 - param_ref['grad_norm_alpha']) * grad_norm
                    )
                    normalized_grad = grad_to_store / max(param_ref['grad_norm_ema'], 1e-8)
                    grad_flat = normalized_grad.view(-1).cpu()
                else:
                    grad_flat = grad_to_store.view(-1).cpu()
            
            param_ref['grad_memory'].append(grad_flat.clone())

    @torch.no_grad()
    def _update_topology(self) -> None:
        """Update eigenspace decomposition with all theoretical fixes."""
        for name, param_ref in self.param_data.items():
            if len(param_ref['grad_memory']) < 3:
                continue
            
            try:
                self._update_parameter_eigenspace(name, param_ref)
            except Exception as e:
                logger.debug(f"Topology update failed for {name}: {e}")

    def step(self, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Perform one optimization step with enhanced stability."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.use_amp:
            with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                output = self.model(x)
                loss = loss_fn(output, y)
        else:
            output = self.model(x)
            loss = loss_fn(output, y)
        
        # Enhanced loss validity check
        if not torch.isfinite(loss) or loss.item() > 1e6:  # Also check for very large losses
            logger.warning(f"Invalid loss detected: {loss.item()}")
            return float('inf'), output
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # Store gradients before unscaling (if not in emergency mode)
            if self.steps % self.store_interval == 0 and not self.emergency_mode:
                self._store_gradients()
            
            # Unscale and clip with more conservative clipping
            self.scaler.unscale_(self.optimizer)
            
            # Check for gradient explosion before clipping
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None and torch.isfinite(p.grad).all():
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > self.gradient_clip_norm * 5:  # Emergency clipping
                logger.warning(f"Large gradient norm detected: {total_norm}")
                self.emergency_mode = True
                clip_norm = self.gradient_clip_norm * 0.1  # Very conservative
            else:
                clip_norm = self.gradient_clip_norm
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            
            # Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Store gradients (if not in emergency mode)
            if self.steps % self.store_interval == 0 and not self.emergency_mode:
                self._store_gradients()
            
            # Conservative clipping for non-AMP
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
        
        # Update state
        self.steps += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self._visualization_data['loss_values'].append(loss_val)
        
        # Update topology less frequently if unstable
        update_freq = self.update_interval * 2 if self.emergency_mode else self.update_interval
        if self.steps % update_freq == 0:
            self._update_topology()
        
        return loss_val, output
    
    def step_complex(self, loss_fn: Callable, batch: Union[torch.Tensor, Dict, Tuple], 
                    y: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """Handle complex input formats."""
        # Handle tuple/list inputs
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            return self.step(loss_fn, inputs, targets)
        
        # Handle dictionary inputs
        elif isinstance(batch, dict):
            self.model.train()
            
            if y is None:
                y = batch.get('labels')
                if y is None:
                    raise ValueError("Dictionary batch must contain 'labels'")
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    outputs = self.model(**batch)
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        loss = loss_fn(logits, y)
            else:
                outputs = self.model(**batch)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    loss = loss_fn(logits, y)
            
            if not torch.isfinite(loss):
                return float('inf'), outputs
            
            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                if self.steps % self.store_interval == 0:
                    self._store_gradients()
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.steps % self.store_interval == 0:
                    self._store_gradients()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            
            # Update state
            self.steps += 1
            loss_val = loss.item()
            self.loss_history.append(loss_val)
            self._visualization_data['loss_values'].append(loss_val)
            
            if self.steps % self.update_interval == 0:
                self._update_topology()
            
            return loss_val, outputs
        
        # Handle single tensor
        else:
            if y is not None:
                return self.step(loss_fn, batch, y)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    @property
    def param_groups(self):
        """Access parameter groups."""
        return self.optimizer.param_groups
    
    def state_dict(self):
        """Get state dict."""
        state = {
            'base_optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'loss_history': self.loss_history[-100:] if self.loss_history else []
        }
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.optimizer.load_state_dict(state_dict['base_optimizer'])
        self.steps = state_dict.get('steps', 0)
        self.loss_history = state_dict.get('loss_history', [])
        if self.scaler is not None and 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler'])
    
    def shutdown(self):
        """Cleanup resources."""
        # Remove hooks
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles.clear()
        
        # Clear data
        self.param_data.clear()
        self._visualization_data.clear()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_visualization_data(self):
        """Get visualization data."""
        return {
            'loss_values': self._visualization_data['loss_values'],
            'loss_history': self.loss_history,
            'bifurcations': self._visualization_data['bifurcations'],
            'eigenvalues_history': self._visualization_data['eigenvalues']
        }
    
    def get_tensorboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for TensorBoard."""
        metrics = {}
        
        # Current eigenvalues
        eigenvalue_data = {}
        for name, param_ref in self.param_data.items():
            if param_ref['eigenvalues'] is not None:
                eigenvalue_data[name] = param_ref['eigenvalues'].detach().cpu().numpy()
        
        if eigenvalue_data:
            metrics['eigenvalues'] = eigenvalue_data
        if self.bifurcations:
            metrics['bifurcations'] = self.bifurcations[-100:]
        
        return metrics
    
    def force_topology_update(self):
        """Force immediate topology update."""
        self._update_topology()

    # Add this diagnostic method to ImprovedTALTOptimizer class
    def diagnose_convergence(self):
        """Diagnose slow convergence issues."""
        print("\n=== TALT Convergence Diagnostics ===")
        
        # Check how many parameters are being tracked
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        tracked_params = len(self.param_data)
        print(f"Tracking {tracked_params} parameters out of {total_params} total")
        
        # Check eigenvalue statistics
        for name, param_ref in self.param_data.items():
            if param_ref['eigenvalues'] is not None:
                eigenvals = param_ref['eigenvalues'].cpu().numpy()
                print(f"\n{name}:")
                print(f"  Eigenvalues range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
                print(f"  Valleys detected: {(eigenvals.abs() < self.valley_threshold).sum()}")
                print(f"  High curvature: {(eigenvals.abs() > 1.0).sum()}")
        
        print(f"\nTotal bifurcations: {len(self.bifurcations)}")
        print(f"Valley threshold: {self.valley_threshold}")
        print(f"Valley amplification: {self.valley_scale}x")