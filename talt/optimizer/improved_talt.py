"""Speed-Optimized TALT Optimizer - Simple, fast, and error-free implementation."""
from collections import deque
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import torch
import torch.nn as nn
import logging
import numpy as np
from torch.cuda.amp import autocast, GradScaler
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
        valley_strength: float = 0.1,
        smoothing_factor: float = 0.3,
        grad_store_interval: int = 5,
        min_param_size: int = 100,
        max_param_size: int = 1000000,
        device: Union[str, torch.device] = "cuda",
        gradient_clip_norm: float = 10.0,
        min_eigenvalue: float = 1e-6,
        regularization_strength: float = 1e-3
    ):
        """Initialize speed-optimized TALT optimizer."""
        self.model = model
        self.optimizer = base_optimizer(model.parameters(), lr=lr)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Initialize GradScaler only for CUDA
        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # TALT parameters
        self.memory_size = memory_size
        self.update_interval = update_interval
        self.valley_strength = valley_strength
        self.smoothing_factor = smoothing_factor
        self.store_interval = grad_store_interval
        self.min_param_size = min_param_size
        self.max_param_size = max_param_size
        
        # Stability parameters
        self.gradient_clip_norm = gradient_clip_norm
        self.min_eigenvalue = min_eigenvalue
        self.regularization_strength = regularization_strength
        
        # State tracking
        self.steps = 0
        self.loss_history = []
        self.bifurcations = []
        
        # Per-parameter state
        self.param_data = {}
        self._hook_handles = {}
        
        # Pre-compute constants
        self.valley_scale = 1.0 + self.valley_strength
        self.valley_threshold = 0.05
        
        # Visualization data
        self._visualization_data = {
            'loss_values': [],
            'bifurcations': [],
            'eigenvalues': {}
        }
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameter tracking."""
        registered_params = 0
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            param_size = param.numel()
            
            if self.min_param_size <= param_size <= self.max_param_size:
                registered_params += 1
                
                # Simple data structure
                self.param_data[name] = {
                    'grad_memory': deque(maxlen=self.memory_size),
                    'principal_dirs': None,
                    'eigenvalues': None,
                    'size': param_size
                }
                
                # Register gradient hook
                handle = param.register_hook(
                    lambda grad, name=name: self._transform_gradient(grad, name)
                )
                self._hook_handles[name] = handle
                
                # Initialize visualization data
                self._visualization_data['eigenvalues'][name] = []
        
        logger.info(f"TALT: Tracking {registered_params} parameters out of {total_params} total")
    
    def _transform_gradient(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """Transform gradient based on eigenspace - simplified for stability."""
        if grad is None or name not in self.param_data:
            return grad
            
        # Check for NaN/Inf
        if not torch.isfinite(grad).all():
            logger.warning(f"Non-finite gradient detected for {name}")
            return torch.zeros_like(grad)
        
        # Gradient clipping
        grad_norm = grad.norm()
        if grad_norm > self.gradient_clip_norm:
            grad = grad * (self.gradient_clip_norm / grad_norm)
        
        param_ref = self.param_data[name]
        dirs = param_ref['principal_dirs']
        vals = param_ref['eigenvalues']
        
        if dirs is None or vals is None:
            return grad
        
        try:
            # Simple transformation without complex conditionals
            flat_grad = grad.view(-1)
            
            # Ensure same device
            if dirs.device != grad.device:
                dirs = dirs.to(grad.device)
                param_ref['principal_dirs'] = dirs
            if vals.device != grad.device:
                vals = vals.to(grad.device)
                param_ref['eigenvalues'] = vals
            
            # Project gradient
            n_components = min(dirs.shape[1], vals.shape[0], self.memory_size)
            dirs_subset = dirs[:, :n_components]
            vals_subset = vals[:n_components]
            
            # Compute coefficients
            coeffs = torch.mv(dirs_subset.t(), flat_grad)
            
            # Apply scaling without complex branching
            # Create scaling factors
            scales = torch.ones_like(vals_subset)
            
            # Valley amplification (low curvature)
            valley_mask = vals_subset.abs() < self.valley_threshold
            scales = torch.where(valley_mask, 
                               torch.tensor(self.valley_scale, device=scales.device), 
                               scales)
            
            # High curvature damping
            high_curve_mask = vals_subset.abs() > 1.0
            damping = self.smoothing_factor / torch.sqrt(vals_subset.abs().clamp(min=self.min_eigenvalue))
            scales = torch.where(high_curve_mask, damping, scales)
            
            # Apply scaling
            coeffs = coeffs * scales
            
            # Project back
            transformed_grad = torch.mv(dirs_subset, coeffs)
            
            # Record bifurcation if valleys detected
            if valley_mask.any():
                self.bifurcations.append(self.steps)
                self._visualization_data['bifurcations'].append(self.steps)
            
            return transformed_grad.view_as(grad)
            
        except Exception as e:
            logger.warning(f"Gradient transformation failed for {name}: {e}")
            return grad
    
    @torch.no_grad()
    def _update_topology(self) -> None:
        """Update eigenspace decomposition for all parameters."""
        for name, param_ref in self.param_data.items():
            if len(param_ref['grad_memory']) < 3:
                continue
                
            try:
                # Stack gradients
                grad_list = list(param_ref['grad_memory'])
                grad_tensor = torch.stack([g.to(self.device) for g in grad_list])
                
                # Skip if low variance
                if grad_tensor.std() < 1e-8:
                    continue
                
                # Center gradients
                mean = grad_tensor.mean(dim=0, keepdim=True)
                centered = grad_tensor - mean
                
                # Compute covariance
                n_samples = centered.shape[0]
                cov = torch.mm(centered.t(), centered) / (n_samples - 1)
                
                # Add regularization
                reg_value = self.regularization_strength * cov.trace() / cov.shape[0]
                cov.diagonal().add_(reg_value)
                
                # Eigendecomposition
                try:
                    eigenvals, eigenvecs = torch.linalg.eigh(cov)
                    
                    # Sort by magnitude
                    idx = eigenvals.abs().argsort(descending=True)[:self.memory_size]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    
                    # Store results
                    param_ref['eigenvalues'] = eigenvals
                    param_ref['principal_dirs'] = eigenvecs
                    
                    # Update visualization
                    top_vals = eigenvals[:3].detach().cpu().numpy() if len(eigenvals) >= 3 else eigenvals.detach().cpu().numpy()
                    self._visualization_data['eigenvalues'][name].append(top_vals)
                    
                except torch.linalg.LinAlgError:
                    logger.warning(f"Eigendecomposition failed for {name}")
                    
            except Exception as e:
                logger.debug(f"Topology update failed for {name}: {e}")
    
    def _store_gradients(self):
        """Store gradients efficiently."""
        for name, param_ref in self.param_data.items():
            # Get parameter by name
            param = None  # Initialize param to None
            for n, p in self.model.named_parameters():
                if n == name:
                    param = p
                    break
            
            if param is None or param.grad is None: # Check if param was found
                continue
            
            grad = param.grad.detach()
            
            # Check validity
            if not torch.isfinite(grad).all():
                continue
            
            # Store as dense tensor
            grad_flat = grad.view(-1).cpu()
            param_ref['grad_memory'].append(grad_flat.clone())
    
    def step(self, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Perform one optimization step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.use_amp:
            with autocast():
                output = self.model(x)
                loss = loss_fn(output, y)
        else:
            output = self.model(x)
            loss = loss_fn(output, y)
        
        # Check loss validity
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss detected")
            return float('inf'), output
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # Store gradients before unscaling
            if self.steps % self.store_interval == 0:
                self._store_gradients()
            
            # Unscale and clip
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            # Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Store gradients
            if self.steps % self.store_interval == 0:
                self._store_gradients()
            
            # Clip and step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
        
        # Update state
        self.steps += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self._visualization_data['loss_values'].append(loss_val)
        
        # Update topology
        if self.steps % self.update_interval == 0:
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
                with autocast():
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