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
        use_power_iteration: bool = True,  # Use faster eigendecomposition for large matrices
        power_iter_steps: int = 5,         # Steps for power iteration
        compile_mode: bool = True          # Use torch.compile for critical functions
    ):
        """Initialize speed-optimized TALT optimizer."""
        self.model = model
        self.optimizer = base_optimizer(model.parameters(), lr=lr)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.scaler = GradScaler()
        
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
        self.loss_history = deque(maxlen=1000)  # Limited history
        self.bifurcations = []
        
        # Per-parameter state with efficient storage
        self.param_data = {}  # Consolidated parameter data
        
        # Pre-compute constants
        self.valley_scale = 1.0 + self.valley_strength
        self.high_curve_threshold = 1.0
        self.low_curve_threshold = 0.5
        self.valley_threshold = 0.05
        
        # Visualization data (minimal for speed)
        self._visualization_data = {
            'loss_values': deque(maxlen=1000),
            'bifurcations': deque(maxlen=100),
            'valley_detections': deque(maxlen=100)
        }
        
        # Initialize parameters efficiently
        self._initialize_parameters()
        
        # Compile critical functions if requested
        if compile_mode and hasattr(torch, 'compile'):
            try:
                self._transform_gradient_core = torch.compile(self._transform_gradient_core)
                self._power_iteration = torch.compile(self._power_iteration)
                logger.info("TALT: Compiled critical functions for speed")
            except:
                logger.warning("TALT: torch.compile not available, running in eager mode")
    
    def _initialize_parameters(self):
        """Efficiently initialize parameter tracking."""
        registered_params = 0
        total_params = 0
        total_model_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Speed-Optimized TALT initialized with {total_model_params} total parameters")
        
        # Pre-allocate all parameter data structures
        for name, param in self.model.named_parameters():
            total_params += 1
            param_size = param.numel()
            
            if (param.requires_grad and 
                param_size >= self.min_param_size and 
                param_size <= self.max_param_size):
                
                registered_params += 1
                
                # Consolidated data structure for efficiency
                self.param_data[name] = {
                    'grad_memory': deque(maxlen=self.memory_size),
                    'principal_dirs': None,
                    'eigenvalues': None,
                    'grad_buffer': torch.zeros(param_size, device='cpu', dtype=param.dtype, pin_memory=True),
                    'transform_matrix': None,  # Cache transformation matrix
                    'last_update_step': -1,    # Track last update
                    'size': param_size,
                    'use_power_iter': param_size > 5000,  # Use power iteration for large params
                    'gradient_norm_history': deque(maxlen=100)
                }
                
                # Register optimized hook
                handle = param.register_hook(
                    lambda grad, name=name, param_ref=self.param_data[name]: 
                    self._fast_transform_gradient(grad, name, param_ref)
                )
                self.param_data[name]['hook_handle'] = handle
        
        logger.info(f"TALT: Tracking {registered_params}/{total_params} parameters")
        
        # Pre-allocate workspace tensors for efficiency
        if registered_params > 0:
            max_param_size = max(pd['size'] for pd in self.param_data.values())
            self._workspace = {
                'coeffs': torch.zeros(self.memory_size, device=self.device),
                'temp_vec': torch.zeros(max_param_size, device=self.device),
                'temp_mat': torch.zeros(self.memory_size, self.memory_size, device=self.device)
            }
    
    @torch.no_grad()
    def _fast_transform_gradient(self, grad: torch.Tensor, name: str, param_ref: Dict) -> torch.Tensor:
        """Fast gradient transformation with minimal overhead."""
        # Quick NaN check
        if not torch.isfinite(grad).all():
            return torch.zeros_like(grad)
        
        # Quick clip
        grad_norm = grad.norm()
        if grad_norm > self.gradient_clip_norm:
            grad.mul_(self.gradient_clip_norm / grad_norm)
        
        # Skip if no eigenspace computed yet
        if param_ref['principal_dirs'] is None:
            return grad
        
        # Use cached transformation if available and recent
        if (param_ref['transform_matrix'] is not None and 
            self.steps - param_ref['last_update_step'] < self.update_interval // 2):
            flat_grad = grad.view(-1)
            transformed = torch.mv(param_ref['transform_matrix'], flat_grad)
            return transformed.view_as(grad)
        
        # Full transformation
        return self._transform_gradient_core(grad, name, param_ref)
    
    def _transform_gradient_core(self, grad: torch.Tensor, name: str, param_ref: Dict) -> torch.Tensor:
        """Core gradient transformation logic."""
        dirs = param_ref['principal_dirs']
        vals = param_ref['eigenvalues']
        
        if dirs is None or vals is None:
            return grad
        
        flat_grad = grad.view(-1)
        
        # Efficient projection using pre-allocated workspace
        n_components = dirs.shape[1]
        coeffs = self._workspace['coeffs'][:n_components]
        
        # Project gradient onto eigenvectors (optimized matmul)
        torch.mv(dirs.t(), flat_grad, out=coeffs)
        
        # Valley detection and scaling (vectorized)
        # Low curvature mask
        low_mask = (vals.abs() < self.valley_threshold) & (vals.abs() > self.min_eigenvalue)
        if low_mask.any():
            self._register_valley_detection(name)
            coeffs[low_mask] *= self.valley_scale
        
        # High curvature scaling (vectorized)
        high_mask = vals.abs() > self.high_curve_threshold
        if high_mask.any():
            # Efficient in-place computation
            scales = self.smoothing_factor / torch.sqrt(vals[high_mask].abs())
            coeffs[high_mask] *= scales
        
        # Moderate curvature boost (vectorized)
        mod_mask = (vals.abs() < self.low_curve_threshold) & ~low_mask
        if mod_mask.any():
            boosts = 1.0 + (1.0 - vals[mod_mask].abs()).clamp(0, 1)
            coeffs[mod_mask] *= boosts
        
        # Efficient back-projection
        temp_vec = self._workspace['temp_vec'][:flat_grad.shape[0]]
        torch.mv(dirs, coeffs, out=temp_vec)
        
        # Cache transformation matrix for reuse
        if param_ref['transform_matrix'] is None:
            param_ref['transform_matrix'] = torch.zeros(dirs.shape[0], dirs.shape[0], device=dirs.device)
        
        # Update cache efficiently
        torch.mm(dirs, torch.diag(coeffs / (torch.mv(dirs.t(), flat_grad) + 1e-8)), out=param_ref['transform_matrix'][:dirs.shape[1], :])
        torch.mm(param_ref['transform_matrix'][:dirs.shape[1], :], dirs.t(), out=param_ref['transform_matrix'])
        param_ref['last_update_step'] = self.steps
        
        return temp_vec.view_as(grad)
    
    def _register_valley_detection(self, name: str):
        """Efficiently register valley detection."""
        self.bifurcations.append(self.steps)
        self._visualization_data['bifurcations'].append(self.steps)
        self._visualization_data['valley_detections'].append((self.steps, name))
    
    @torch.no_grad()
    def _update_topology_batch(self) -> None:
        """Batch topology update for all parameters (more efficient)."""
        updates_needed = []
        
        # Collect parameters needing updates
        for name, param_ref in self.param_data.items():
            if len(param_ref['grad_memory']) >= 3:
                updates_needed.append((name, param_ref))
        
        if not updates_needed:
            return
        
        # Process updates in parallel where possible
        for name, param_ref in updates_needed:
            try:
                # Convert gradients efficiently
                grad_tensor = torch.stack([g.to(self.device) if not g.is_sparse else g.to_dense().to(self.device) 
                                          for g in param_ref['grad_memory']])
                
                # Skip if low variance
                if grad_tensor.std() < 1e-8:
                    continue
                
                # Efficient eigendecomposition based on size
                if param_ref['use_power_iter'] and param_ref['size'] > 5000:
                    # Use power iteration for large parameters
                    eigenvals, eigenvecs = self._power_iteration(grad_tensor, param_ref)
                else:
                    # Standard method for smaller parameters
                    eigenvals, eigenvecs = self._efficient_eigen_decomposition(grad_tensor)
                
                if eigenvals is not None and eigenvecs is not None:
                    param_ref['eigenvalues'] = eigenvals
                    param_ref['principal_dirs'] = eigenvecs
                    param_ref['transform_matrix'] = None  # Reset cache
                    
                    # Store minimal visualization data
                    if hasattr(self, '_visualization_data'):
                        self._update_viz_data(name, eigenvals, grad_tensor)
                        
            except Exception as e:
                logger.debug(f"Topology update failed for {name}: {e}")
    
    def _efficient_eigen_decomposition(self, grad_tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Efficient eigendecomposition with stability."""
        n_samples = grad_tensor.shape[0]
        
        # Center gradients
        mean = grad_tensor.mean(dim=0, keepdim=True)
        centered = grad_tensor - mean
        
        # Efficient covariance computation
        cov = torch.mm(centered.t(), centered) / (n_samples - 1)
        
        # Adaptive regularization (fast)
        trace = cov.diagonal().sum()
        reg_strength = max(self.regularization_strength, 0.01 * trace / cov.shape[0])
        cov.diagonal().add_(reg_strength)
        
        try:
            # Use symeig for symmetric matrices (faster than eigh)
            eigenvals, eigenvecs = torch.linalg.eigh(cov)
            
            # Sort and select top components
            idx = eigenvals.abs().argsort(descending=True)[:self.memory_size]
            eigenvals = eigenvals[idx].clamp(min=self.min_eigenvalue)
            eigenvecs = eigenvecs[:, idx]
            
            return eigenvals, eigenvecs
            
        except:
            return None, None
    
    def _power_iteration(self, grad_tensor: torch.Tensor, param_ref: Dict) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Fast power iteration for top eigenvalues/vectors."""
        n_samples, n_features = grad_tensor.shape
        k = min(self.memory_size, n_features, 10)  # Limit components
        
        # Center gradients
        mean = grad_tensor.mean(dim=0, keepdim=True)
        centered = grad_tensor - mean
        
        # Initialize random vectors
        V = torch.randn(n_features, k, device=grad_tensor.device)
        V = torch.qr(V).Q  # Orthogonalize
        
        eigenvals = torch.zeros(k, device=grad_tensor.device)
        
        # Power iteration
        for _ in range(self.power_iter_steps):
            # Efficient matrix multiplication
            AV = torch.mm(centered.t(), torch.mm(centered, V)) / (n_samples - 1)
            
            # QR decomposition for stability
            V, R = torch.qr(AV)
            
            # Extract eigenvalues from diagonal of R
            eigenvals = R.diagonal()
        
        # Final eigenvalues
        eigenvals = eigenvals.abs().clamp(min=self.min_eigenvalue)
        
        return eigenvals, V
    
    def _update_viz_data(self, name: str, eigenvals: torch.Tensor, grad_tensor: torch.Tensor):
        """Minimal visualization data update."""
        if name not in self._visualization_data:
            self._visualization_data[name] = {
                'eigenvalues': deque(maxlen=50),
                'gradient_stats': deque(maxlen=50)
            }
        
        # Store only essential data
        top_eigenvals = eigenvals[:3].detach().cpu().numpy() if len(eigenvals) >= 3 else eigenvals.detach().cpu().numpy()
        grad_norm = grad_tensor[-1].norm().item()
        
        self._visualization_data[name]['eigenvalues'].append((self.steps, top_eigenvals))
        self._visualization_data[name]['gradient_stats'].append({
            'step': self.steps,
            'grad_norm': grad_norm
        })
    
    @torch.no_grad()
    def step(self, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Optimized TALT step."""
        self.optimizer.zero_grad(set_to_none=True)  # More efficient
        
        # Forward pass
        with autocast('cuda'):
            output = self.model(x)
            loss = loss_fn(output, y)
        
        # Quick NaN check
        if not torch.isfinite(loss):
            return 1e6, output
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Store gradients efficiently
        if self.steps % self.store_interval == 0:
            self._efficient_grad_storage()
        
        # Gradient clipping (in-place)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        
        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
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
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self.param_data:
                continue
            
            param_ref = self.param_data[name]
            grad = param.grad
            
            # Quick finite check
            if not torch.isfinite(grad).all():
                continue
            
            # Efficient copy to pinned memory
            grad_flat = grad.view(-1)
            
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
            param_ref['grad_buffer'].copy_(grad_flat, non_blocking=True)
            param_ref['grad_memory'].append(param_ref['grad_buffer'].clone())
            
            # Track gradient norm
            param_ref['gradient_norm_history'].append(grad_flat.norm().item())
    
    def step_complex(self, loss_fn: Callable, batch: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple], 
                    y: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """Optimized complex step for different input formats."""
        # Fast path for common cases
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return self.step(loss_fn, batch[0].to(self.device, non_blocking=True), 
                           batch[1].to(self.device, non_blocking=True))
        
        # Dictionary inputs (transformers)
        if isinstance(batch, dict):
            if y is None:
                y = batch.get('labels')
            
            # Move to device efficiently
            inputs = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = loss_fn(logits, y.to(self.device, non_blocking=True))
            
            if not torch.isfinite(loss):
                return 1e6, logits
            
            self.scaler.scale(loss).backward()
            
            # Efficient gradient operations
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            if self.steps % self.store_interval == 0:
                self._efficient_grad_storage()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.steps += 1
            loss_val = loss.item()
            self.loss_history.append(loss_val)
            
            if self.steps % self.update_interval == 0:
                self._update_topology_batch()
            
            return loss_val, logits
        
        # Default case
        return self.step(loss_fn, batch, y)
    
    def zero_grad(self, set_to_none: bool = True):
        """Efficient gradient zeroing."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    @property
    def param_groups(self):
        """Access parameter groups."""
        return self.optimizer.param_groups
    
    def state_dict(self):
        """Minimal state dict for fast checkpointing."""
        return {
            'base_optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'loss_history': list(self.loss_history)[-100:]  # Keep only recent
        }
    
    def load_state_dict(self, state_dict):
        """Fast state loading."""
        self.optimizer.load_state_dict(state_dict['base_optimizer'])
        self.steps = state_dict['steps']
        self.loss_history = deque(state_dict.get('loss_history', []), maxlen=1000)
    
    def shutdown(self):
        """Efficient cleanup."""
        # Clear large data structures
        self.param_data.clear()
        self._workspace.clear() if hasattr(self, '_workspace') else None
        self._visualization_data.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_visualization_data(self):
        """Get minimal visualization data efficiently."""
        # Consolidate visualization data
        viz_data = {
            'loss_values': list(self._visualization_data['loss_values']),
            'loss_history': list(self.loss_history),
            'bifurcations': list(self._visualization_data['bifurcations']),
            'valley_detections': list(self._visualization_data['valley_detections']),
            'eigenvalues_history': {},
            'gradient_norms_history': {},
            'gradient_stats': {}
        }
        
        # Efficiently collect parameter-specific data
        for name, param_ref in self.param_data.items():
            if name in self._visualization_data:
                param_viz = self._visualization_data[name]
                
                # Eigenvalues
                if 'eigenvalues' in param_viz and param_viz['eigenvalues']:
                    steps, eigenvals = zip(*list(param_viz['eigenvalues']))
                    viz_data['eigenvalues_history'][name] = {
                        'steps': list(steps),
                        'eigenvalues': list(eigenvals)
                    }
                
                # Gradient stats
                if 'gradient_stats' in param_viz:
                    viz_data['gradient_stats'][name] = list(param_viz['gradient_stats'])
            
            # Gradient norms
            if param_ref['gradient_norm_history']:
                viz_data['gradient_norms_history'][name] = {
                    'grad_norms': list(param_ref['gradient_norm_history']),
                    'steps': list(range(len(param_ref['gradient_norm_history'])))
                }
        
        return viz_data
    
    def get_tensorboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for TensorBoard efficiently."""
        metrics = {}
        
        # Collect current metrics with minimal overhead
        eigenvalue_data = {}
        gradient_norms = {}
        
        for name, param_ref in self.param_data.items():
            if param_ref['eigenvalues'] is not None:
                eigenvalue_data[name] = param_ref['eigenvalues'].detach().cpu().numpy()
            
            if param_ref['gradient_norm_history']:
                gradient_norms[name] = param_ref['gradient_norm_history'][-1]
        
        if eigenvalue_data:
            metrics['eigenvalues'] = eigenvalue_data
        if gradient_norms:
            metrics['gradient_norms'] = gradient_norms
        if self.bifurcations:
            metrics['bifurcations'] = self.bifurcations[-100:]  # Recent only
        
        return metrics
    
    def diagnose_visualization_state(self):
        """Quick diagnostic output."""
        logger.info(f"TALT Diagnostics: steps={self.steps}, loss_history={len(self.loss_history)}, "
                   f"params={len(self.param_data)}, bifurcations={len(self.bifurcations)}")
    
    def force_topology_update(self):
        """Force immediate topology update."""
        self._update_topology_batch()