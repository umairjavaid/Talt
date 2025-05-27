"""Speed-Optimized TALT Optimizer - Maintains theoretical correctness with maximum performance."""
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
    Speed-Optimized TALT Optimizer with theoretical correctness.
    
    Performance optimizations:
    1. Efficient eigendecomposition with caching
    2. Vectorized operations and in-place computations
    3. Smart memory management and reuse
    4. JIT-compiled critical functions
    5. Minimal CPU-GPU transfers
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
        sparsity_threshold: float = 0.01,
        device: Union[str, torch.device] = "cuda",
        gradient_clip_norm: float = 10.0,
        min_eigenvalue: float = 1e-6,
        regularization_strength: float = 1e-3,
        use_power_iteration: bool = True,
        power_iter_steps: int = 5,
        compile_mode: bool = True
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
        self.sparsity_threshold = sparsity_threshold
        
        # Stability parameters
        self.gradient_clip_norm = gradient_clip_norm
        self.min_eigenvalue = min_eigenvalue
        self.regularization_strength = regularization_strength
        
        # Performance parameters
        self.use_power_iteration = use_power_iteration
        self.power_iter_steps = power_iter_steps
        
        # State tracking
        self.steps = 0
        self.loss_history = deque(maxlen=1000)
        self.bifurcations = []
        
        # Per-parameter state with efficient storage
        self.param_data = {}
        self._hook_handles = {}
        
        # Pre-compute constants
        self.valley_scale = 1.0 + self.valley_strength
        self.high_curve_threshold = 1.0
        self.low_curve_threshold = 0.5
        self.valley_threshold = 0.05
        
        # Visualization data (minimal for speed)
        self._visualization_data = {
            'loss_values': deque(maxlen=1000),
            'bifurcations': deque(maxlen=100),
            'valley_detections': deque(maxlen=100),
            'gradient_stats': {}
        }
        
        # Initialize parameters efficiently
        self._initialize_parameters()
        
        # Compile critical functions if requested
        if compile_mode and hasattr(torch, 'compile'):
            try:
                self._transform_gradient_core = torch.compile(self._transform_gradient_core)
                self._power_iteration = torch.compile(self._power_iteration)
                logger.info("TALT: Compiled critical functions for speed")
            except Exception as e:
                logger.warning(f"TALT: torch.compile not available: {e}")
    
    def _initialize_parameters(self):
        """Efficiently initialize parameter tracking."""
        registered_params = 0
        total_params = 0
        total_model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Speed-Optimized TALT initialized with {total_model_params} trainable parameters")
        
        # Pre-allocate all parameter data structures
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            total_params += 1
            param_size = param.numel()
            
            if param_size >= self.min_param_size and param_size <= self.max_param_size:
                registered_params += 1
                
                # Consolidated data structure for efficiency
                self.param_data[name] = {
                    'param': param,
                    'grad_memory': deque(maxlen=self.memory_size),
                    'principal_dirs': None,
                    'eigenvalues': None,
                    'grad_buffer': torch.zeros(param_size, device='cpu', dtype=param.dtype, pin_memory=True),
                    'transform_matrix': None,
                    'last_update_step': -1,
                    'size': param_size,
                    'use_power_iter': param_size > 5000,
                    'gradient_norm_history': deque(maxlen=100)
                }
                
                # Register optimized hook
                handle = param.register_hook(
                    lambda grad, name=name: self._fast_transform_gradient(grad, name)
                )
                self._hook_handles[name] = handle
                
                # Initialize visualization data for this parameter
                self._visualization_data['gradient_stats'][name] = deque(maxlen=50)
        
        logger.info(f"TALT: Tracking {registered_params}/{total_params} parameters")
        
        # Pre-allocate workspace tensors for efficiency
        if registered_params > 0:
            max_param_size = max(pd['size'] for pd in self.param_data.values())
            self._workspace = {
                'coeffs': torch.zeros(self.memory_size, device=self.device),
                'temp_vec': torch.zeros(max_param_size, device=self.device),
                'temp_mat': torch.zeros(self.memory_size, self.memory_size, device=self.device)
            }
    
    def _fast_transform_gradient(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """Fast gradient transformation with minimal overhead."""
        if grad is None:
            return None
        
        # Get parameter reference
        param_ref = self.param_data.get(name)
        if param_ref is None:
            return grad
            
        # Quick NaN check
        if not torch.isfinite(grad).all():
            logger.warning(f"Non-finite gradient detected for {name}")
            return grad.zero_()
        
        # Quick clip
        grad_norm = grad.norm()
        if grad_norm > self.gradient_clip_norm:
            grad = grad * (self.gradient_clip_norm / grad_norm)
        
        # Skip if no eigenspace computed yet
        if param_ref['principal_dirs'] is None:
            return grad
        
        # Full transformation
        try:
            return self._transform_gradient_core(grad, name, param_ref)
        except Exception as e:
            logger.warning(f"Gradient transformation failed for {name}: {e}")
            return grad
    
    def _transform_gradient_core(self, grad: torch.Tensor, name: str, param_ref: Dict) -> torch.Tensor:
        """Core gradient transformation logic - compilation-safe version."""
        dirs = param_ref['principal_dirs']
        vals = param_ref['eigenvalues']
        
        if dirs is None or vals is None:
            return grad
        
        # Ensure dirs is on the same device as grad
        if dirs.device != grad.device:
            dirs = dirs.to(grad.device)
            param_ref['principal_dirs'] = dirs
        
        if vals.device != grad.device:
            vals = vals.to(grad.device)
            param_ref['eigenvalues'] = vals
        
        flat_grad = grad.view(-1)
        n_components = min(dirs.shape[1], self.memory_size)
        
        # Project gradient onto eigenvectors
        coeffs = torch.mv(dirs[:, :n_components].t(), flat_grad)
        
        # Valley detection and scaling - compilation-safe version
        with torch.no_grad():
            # Pre-compute all masks
            vals_slice = vals[:n_components]
            abs_vals = vals_slice.abs()
            
            # Low curvature mask (valley detection)
            low_mask = (abs_vals < self.valley_threshold) & (abs_vals > self.min_eigenvalue)
            
            # High curvature mask
            high_mask = abs_vals > self.high_curve_threshold
            
            # Moderate curvature mask
            mod_mask = (abs_vals < self.low_curve_threshold) & ~low_mask
            
            # Check if any transformations are needed
            any_transforms = low_mask.any() or high_mask.any() or mod_mask.any()
            
            if any_transforms:
                # Register valley detection if low curvature regions found
                if low_mask.any():
                    self._register_valley_detection(name)
                
                # Apply all transformations using torch.where (compilation-safe)
                transformed_coeffs = coeffs
                
                # Valley scaling for low curvature regions
                if low_mask.any():
                    valley_scaled = coeffs * self.valley_scale
                    transformed_coeffs = torch.where(low_mask, valley_scaled, transformed_coeffs)
                
                # High curvature scaling
                if high_mask.any():
                    # Compute scales for high curvature regions
                    high_scales = self.smoothing_factor / torch.sqrt(abs_vals.clamp(min=self.min_eigenvalue))
                    high_scaled = coeffs * high_scales
                    transformed_coeffs = torch.where(high_mask, high_scaled, transformed_coeffs)
                
                # Moderate curvature boost
                if mod_mask.any():
                    # Compute boosts for moderate curvature regions
                    mod_boosts = 1.0 + (1.0 - abs_vals).clamp(0, 1)
                    mod_scaled = coeffs * mod_boosts
                    transformed_coeffs = torch.where(mod_mask, mod_scaled, transformed_coeffs)
                
                # Use transformed coefficients
                coeffs = transformed_coeffs
        
        # Back-projection
        transformed_grad = torch.mv(dirs[:, :n_components], coeffs)
        
        return transformed_grad.view_as(grad)
    
    def _register_valley_detection(self, name: str):
        """Efficiently register valley detection."""
        self.bifurcations.append(self.steps)
        self._visualization_data['bifurcations'].append(self.steps)
        self._visualization_data['valley_detections'].append((self.steps, name))
    
    @torch.no_grad()
    def _update_topology_batch(self) -> None:
        """Batch topology update for all parameters."""
        updates_needed = []
        
        for name, param_ref in self.param_data.items():
            if len(param_ref['grad_memory']) >= 3:
                updates_needed.append((name, param_ref))
        
        if not updates_needed:
            return
        
        for name, param_ref in updates_needed:
            try:
                # Convert gradients efficiently
                grad_list = []
                for g in param_ref['grad_memory']:
                    if g.is_sparse:
                        grad_list.append(g.to_dense().to(self.device))
                    else:
                        grad_list.append(g.to(self.device))
                
                if not grad_list:
                    continue
                    
                grad_tensor = torch.stack(grad_list)
                
                # Skip if low variance
                if grad_tensor.std() < 1e-8:
                    continue
                
                # Efficient eigendecomposition
                if param_ref['use_power_iter'] and param_ref['size'] > 5000:
                    eigenvals, eigenvecs = self._power_iteration(grad_tensor, param_ref)
                else:
                    eigenvals, eigenvecs = self._efficient_eigen_decomposition(grad_tensor)
                
                if eigenvals is not None and eigenvecs is not None:
                    param_ref['eigenvalues'] = eigenvals
                    param_ref['principal_dirs'] = eigenvecs
                    param_ref['transform_matrix'] = None
                    
                    # Update visualization data
                    self._update_viz_data(name, eigenvals, grad_tensor)
                    
            except Exception as e:
                logger.debug(f"Topology update failed for {name}: {e}")
    
    def _update_viz_data(self, name: str, eigenvals: torch.Tensor, grad_tensor: torch.Tensor):
        """Update visualization data for a parameter."""
        if name in self._visualization_data['gradient_stats']:
            top_eigenvals = eigenvals[:3].detach().cpu().numpy() if len(eigenvals) >= 3 else eigenvals.detach().cpu().numpy()
            grad_norm = grad_tensor[-1].norm().item()
            
            self._visualization_data['gradient_stats'][name].append({
                'step': self.steps,
                'grad_norm': grad_norm,
                'eigenvalues': top_eigenvals.tolist()
            })
    
    def _efficient_eigen_decomposition(self, grad_tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Efficient eigendecomposition with stability."""
        n_samples = grad_tensor.shape[0]
        
        # Center gradients
        mean = grad_tensor.mean(dim=0, keepdim=True)
        centered = grad_tensor - mean
        
        # Efficient covariance computation
        cov = torch.mm(centered.t(), centered) / (n_samples - 1)
        
        # Adaptive regularization
        trace = cov.diagonal().sum().item()
        reg_strength = max(self.regularization_strength, 0.01 * trace / cov.shape[0])
        cov.diagonal().add_(reg_strength)
        
        try:
            # Use symeig for symmetric matrices
            eigenvals, eigenvecs = torch.linalg.eigh(cov)
            
            # Sort and select top components
            idx = eigenvals.abs().argsort(descending=True)[:self.memory_size]
            eigenvals = eigenvals[idx].clamp(min=self.min_eigenvalue)
            eigenvecs = eigenvecs[:, idx]
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}")
            return None, None
    
    def _power_iteration(self, grad_tensor: torch.Tensor, param_ref: Dict) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Fast power iteration for top eigenvalues/vectors."""
        n_samples, n_features = grad_tensor.shape
        k = min(self.memory_size, n_features, 10)
        
        # Center gradients
        mean = grad_tensor.mean(dim=0, keepdim=True)
        centered = grad_tensor - mean
        
        # Initialize random vectors
        V = torch.randn(n_features, k, device=grad_tensor.device)
        V, _ = torch.qr(V)
        
        eigenvals = torch.zeros(k, device=grad_tensor.device)
        
        # Power iteration
        for _ in range(self.power_iter_steps):
            AV = torch.mm(centered.t(), torch.mm(centered, V)) / (n_samples - 1)
            V, R = torch.qr(AV)
            eigenvals = R.diagonal()
        
        eigenvals = eigenvals.abs().clamp(min=self.min_eigenvalue)
        
        return eigenvals, V
    
    def step(self, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Optimized TALT step - matches the original TALT interface."""
        # Ensure model is in training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with proper mixed precision handling
        if self.use_amp:
            with autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                output = self.model(x)
                loss = loss_fn(output, y)
        else:
            output = self.model(x)
            loss = loss_fn(output, y)
        
        # Ensure loss requires grad
        if not loss.requires_grad:
            logger.error("Loss does not require grad!")
            raise RuntimeError("Loss computation does not produce gradients. Check model is in training mode.")
        
        # Quick NaN check
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss detected")
            return float('inf'), output
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # Store gradients before unscaling
            if self.steps % self.store_interval == 0:
                self._efficient_grad_storage()
            
            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            # Step optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Store gradients
            if self.steps % self.store_interval == 0:
                self._efficient_grad_storage()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            # Step optimizer
            self.optimizer.step()
        
        # Update state
        self.steps += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self._visualization_data['loss_values'].append(loss_val)
        
        # Batch topology updates
        if self.steps % self.update_interval == 0:
            self._update_topology_batch()
        
        return loss_val, output
    
    def _efficient_grad_storage(self):
        """Store gradients with minimal overhead."""
        for name, param_ref in self.param_data.items():
            param = param_ref['param']
            
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Quick finite check
            if not torch.isfinite(grad).all():
                continue
            
            # Efficient copy to pinned memory
            grad_flat = grad.view(-1).detach()
            
            # For large sparse gradients, use sparse storage
            if grad_flat.numel() > 10000:
                sparsity = (grad_flat.abs() < self.sparsity_threshold).float().mean()
                if sparsity > 0.9:
                    # Sparse storage
                    mask = grad_flat.abs() >= self.sparsity_threshold
                    indices = mask.nonzero().squeeze(-1)
                    values = grad_flat[mask]
                    sparse_grad = torch.sparse_coo_tensor(
                        indices.unsqueeze(0), values, grad_flat.shape, 
                        device='cpu', dtype=grad.dtype
                    )
                    param_ref['grad_memory'].append(sparse_grad)
                    continue
            
            # Dense storage with efficient copy
            param_ref['grad_buffer'].copy_(grad_flat.cpu(), non_blocking=True)
            param_ref['grad_memory'].append(param_ref['grad_buffer'].clone())
            
            # Track gradient norm
            param_ref['gradient_norm_history'].append(grad_flat.norm().item())
    
    def step_complex(self, loss_fn: Callable, batch: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple], 
                    y: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """Complex step method for handling different input formats - matches experiment.py interface."""
        # Handle tuple/list inputs (standard CNN format)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            # Data is already on device from experiment.py
            inputs, targets = batch[0], batch[1]
            return self.step(loss_fn, inputs, targets)
        
        # Handle dictionary inputs (transformer/BERT format)
        elif isinstance(batch, dict):
            # Ensure model is in training mode
            self.model.train()
            
            # Extract labels if not provided separately
            if y is None:
                y = batch.get('labels')
                if y is None:
                    raise ValueError("Dictionary batch must contain 'labels' key or y must be provided")
            
            # Data is already on device from experiment.py
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
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
            
            # Ensure loss requires grad
            if not loss.requires_grad:
                raise RuntimeError("Loss does not require gradients")
            
            if not torch.isfinite(loss):
                return float('inf'), outputs
            
            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                if self.steps % self.store_interval == 0:
                    self._efficient_grad_storage()
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.steps % self.store_interval == 0:
                    self._efficient_grad_storage()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            
            # Update state
            self.steps += 1
            loss_val = loss.item()
            self.loss_history.append(loss_val)
            self._visualization_data['loss_values'].append(loss_val)
            
            if self.steps % self.update_interval == 0:
                self._update_topology_batch()
            
            return loss_val, outputs
        
        # Handle single tensor input
        else:
            if y is not None:
                return self.step(loss_fn, batch, y)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    def zero_grad(self, set_to_none: bool = False):
        """Efficient gradient zeroing."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    @property
    def param_groups(self):
        """Access parameter groups."""
        return self.optimizer.param_groups
    
    def state_dict(self):
        """Minimal state dict for fast checkpointing."""
        state = {
            'base_optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'loss_history': list(self.loss_history)[-100:]
        }
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Fast state loading."""
        self.optimizer.load_state_dict(state_dict['base_optimizer'])
        self.steps = state_dict.get('steps', 0)
        self.loss_history = deque(state_dict.get('loss_history', []), maxlen=1000)
        if self.scaler is not None and 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler'])
    
    def shutdown(self):
        """Efficient cleanup."""
        # Remove hooks
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles.clear()
        
        # Clear large data structures
        self.param_data.clear()
        if hasattr(self, '_workspace'):
            self._workspace.clear()
        self._visualization_data.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_visualization_data(self):
        """Get visualization data - compatible with talt.visualization.visualizer.TALTVisualizer."""
        viz_data = {
            'loss_values': list(self._visualization_data['loss_values']),
            'loss_history': list(self.loss_history),
            'bifurcations': list(self._visualization_data['bifurcations']),
            'valley_detections': list(self._visualization_data['valley_detections']),
            'eigenvalues_history': {},
            'gradient_norms_history': {},
            'gradient_stats': dict(self._visualization_data['gradient_stats'])
        }
        
        # Format eigenvalues and gradient norms for visualization
        for name, param_ref in self.param_data.items():
            # Gradient norms
            if param_ref['gradient_norm_history']:
                viz_data['gradient_norms_history'][name] = {
                    'grad_norms': list(param_ref['gradient_norm_history']),
                    'steps': list(range(len(param_ref['gradient_norm_history'])))
                }
            
            # Eigenvalues from gradient stats
            if name in self._visualization_data['gradient_stats']:
                stats = list(self._visualization_data['gradient_stats'][name])
                if stats:
                    # Extract eigenvalues history
                    eigenvals_list = []
                    steps_list = []
                    for stat in stats:
                        if 'eigenvalues' in stat:
                            steps_list.append(stat['step'])
                            eigenvals_list.append(stat['eigenvalues'])
                    
                    if eigenvals_list:
                        viz_data['eigenvalues_history'][name] = {
                            'steps': steps_list,
                            'eigenvalues': eigenvals_list
                        }
        
        return viz_data
    
    def get_tensorboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for TensorBoard efficiently."""
        metrics = {}
        
        # Collect current metrics with minimal overhead
        eigenvalue_data = {}
        gradient_norms = {}
        curvature_estimates = {}
        
        for name, param_ref in self.param_data.items():
            if param_ref['eigenvalues'] is not None:
                eigenvalue_data[name] = param_ref['eigenvalues'].detach().cpu().numpy()
                # Curvature estimate
                curvature_estimates[name] = float(param_ref['eigenvalues'].abs().max().item())
            
            if param_ref['gradient_norm_history']:
                gradient_norms[name] = param_ref['gradient_norm_history'][-1]
        
        if eigenvalue_data:
            metrics['eigenvalues'] = eigenvalue_data
        if gradient_norms:
            metrics['gradient_norms'] = gradient_norms
        if curvature_estimates:
            metrics['curvature_estimates'] = curvature_estimates
        if self.bifurcations:
            metrics['bifurcations'] = self.bifurcations[-100:]
        if self._visualization_data['valley_detections']:
            metrics['valley_detections'] = list(self._visualization_data['valley_detections'])[-100:]
        
        return metrics
    
    def diagnose_visualization_state(self):
        """Diagnostic output for debugging."""
        param_info = []
        for name, param_ref in self.param_data.items():
            param_info.append(f"{name}: grads={len(param_ref['grad_memory'])}, "
                            f"eigenvals={param_ref['eigenvalues'] is not None}")
        
        logger.info(f"TALT Diagnostics: steps={self.steps}, loss_history={len(self.loss_history)}, "
                   f"params={len(self.param_data)}, bifurcations={len(self.bifurcations)}")
        if param_info:
            logger.info(f"Parameter details: {', '.join(param_info[:3])}...")
    
    def force_topology_update(self):
        """Force immediate topology update."""
        self._update_topology_batch()