"""Main implementation of the Original TALT Optimizer."""

import torch
import torch.nn as nn
import logging
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from torch.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class TALTOptimizer:
    """
    Original TALT Optimizer - Exact Theoretical Implementation
    
    This implements the pure TALT algorithm without approximations:
    - Full gradient dimensionality (no random projection)
    - Direct eigendecomposition using torch.linalg.eigh
    - Batch covariance computation
    - Exact eigenspace projection and reconstruction
    
    Mathematical Algorithm:
    1. Store gradients: G = [g₁, g₂, ..., gₖ]
    2. Center: Ḡ = G - mean(G)  
    3. Covariance: C = ḠᵀḠ/(k-1)
    4. Eigendecomposition: C = VΛVᵀ
    5. Project: coeffs = gᵀV
    6. Transform coeffs based on eigenvalues
    7. Reconstruct: g_new = V * coeffs_modified
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: Callable,  # e.g., optim.SGD, optim.Adam
        *,
        lr: float = 1e-2,
        eigenspace_memory_size: int = 10,
        topology_update_interval: int = 20,
        valley_strength: float = 0.1,
        smoothing_factor: float = 0.3,
        grad_store_interval: int = 5,
        min_param_size: int = 10,
        device: Union[str, torch.device] = "cuda"
    ):
        # Core setup
        self.model = model
        self.optimizer = base_optimizer(model.parameters(), lr=lr)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.scaler = GradScaler()
        
        # TALT parameters
        self.memory_size = eigenspace_memory_size
        self.update_interval = topology_update_interval
        self.valley_strength = valley_strength
        self.smoothing_factor = smoothing_factor
        self.store_interval = grad_store_interval
        self.min_param_size = min_param_size
        
        # State tracking
        self.steps = 0
        self.loss_history = []
        self.bifurcations = []  # Valley detection points
        
        # Async topology updates
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._topology_update_future = None
        
        # Per-parameter gradient tracking
        self.grad_memory = {}      # Recent gradients buffer
        self.principal_dirs = {}   # Eigenvectors V
        self.eigenvalues = {}      # Eigenvalues Λ
        
        # Visualization data
        self._visualization_data = {
            'loss_values': [],
            'bifurcation_points': [],
            'eigenvalues': {}
        }
        
        # Register gradient hooks for parameters above size threshold
        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > self.min_param_size:
                self.grad_memory[name] = deque(maxlen=self.memory_size)
                self.principal_dirs[name] = None
                self.eigenvalues[name] = None
                self._visualization_data['eigenvalues'][name] = []
                
                # Hook transforms gradients using eigenspace analysis
                param.register_hook(lambda grad, name=name: self._transform_gradient(grad, name))

    def _transform_gradient(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """
        CORE TALT ALGORITHM: Transform gradients using eigenspace analysis
        
        Steps:
        1. Project gradient onto eigenvectors: coeffs = grad^T * V
        2. Identify valley directions: |eigenvalue| < valley_threshold  
        3. Amplify valley coefficients: coeffs[valleys] *= (1 + valley_strength)
        4. Dampen high-curvature coefficients: coeffs[i] *= smoothing_factor/√|λᵢ|
        5. Project back to gradient space: grad_new = V * coeffs_modified
        """
        if name not in self.grad_memory or grad.numel() < self.min_param_size:
            return grad
            
        # Store original for fallback
        orig_grad = grad.clone()
        dirs = self.principal_dirs[name]  # Eigenvectors V
        vals = self.eigenvalues[name]     # Eigenvalues Λ
        valley_strength = self.valley_strength
        smoothing_factor = self.smoothing_factor
            
        if dirs is None or vals is None:
            return grad  # No eigenspace info yet
            
        try:
            # Step 1: Project gradient onto eigenspace
            flat_grad = grad.view(-1)
            dirs_device = dirs.to(self.device)
            coeffs = torch.matmul(flat_grad, dirs_device)  # coeffs = g^T * V
            
            # Step 2: Identify valley directions (low curvature)
            low_curvature_mask = (vals.abs() < 0.05)
            
            # Step 3: Record bifurcation and amplify valleys
            if low_curvature_mask.any():
                self.bifurcations.append(self.steps)
                # Amplify movement in valley directions
                coeffs[low_curvature_mask] *= (1.0 + valley_strength)
            
            # Step 4: Apply curvature-based coefficient scaling
            for i, eigenval in enumerate(vals):
                abs_val = abs(eigenval)
                if abs_val > 1.0:
                    # High curvature: dampen to avoid oscillations
                    scale = (1.0 / torch.sqrt(torch.tensor(abs_val))) * smoothing_factor
                    coeffs[i] *= scale
                elif abs_val < 0.5:
                    # Low curvature: boost for faster convergence
                    coeffs[i] *= (1.0 + (1.0 - abs_val))
            
            # Step 5: Project back to parameter space
            transformed_grad = torch.matmul(coeffs, dirs_device.t()).view_as(grad)
            
            # Safety: Check transformation quality
            cos_sim = nn.functional.cosine_similarity(
                transformed_grad.view(-1), orig_grad.view(-1), dim=0
            )
            
            # Blend with original if transformation too aggressive
            if cos_sim < 0.7:
                blend_factor = 0.7 - cos_sim
                transformed_grad = (1.0 - blend_factor) * transformed_grad + blend_factor * orig_grad
                
            # Safety: Check for NaN/Inf
            if torch.isnan(transformed_grad).any() or torch.isinf(transformed_grad).any():
                logger.warning(f"NaN/Inf in transformed gradient for {name}")
                return orig_grad
                
            return transformed_grad
            
        except Exception as e:
            logger.warning(f"TALT gradient transformation error for {name}: {e}")
            return orig_grad

    def _update_topology(self) -> None:
        """
        Update eigenspace decomposition of gradient covariance matrices.
        
        For each parameter:
        1. Stack recent gradients into matrix G
        2. Center gradients: Ḡ = G - mean(G)
        3. Compute covariance: C = ḠᵀḠ/(n-1)
        4. Eigendecomposition: C = VΛVᵀ
        5. Store V (eigenvectors) and Λ (eigenvalues)
        """
        for name, grad_buffer in self.grad_memory.items():
            if len(grad_buffer) < 2:
                continue
                
            try:
                # Stack gradients into matrix [n_grads, param_dim]
                stacked = torch.stack(list(grad_buffer))
                
                # Center gradients (subtract mean)
                centered = stacked - stacked.mean(dim=0, keepdim=True)
                
                # Compute covariance matrix
                cov = torch.matmul(centered.t(), centered) / (stacked.size(0) - 1)
                
                # Eigendecomposition (CPU for stability)
                cpu_cov = cov.cpu()
                eigenvals, eigenvecs = torch.linalg.eigh(cpu_cov)
                
                # Sort by absolute eigenvalue (descending)
                idx = torch.argsort(-eigenvals.abs())
                eigenvecs, eigenvals = eigenvecs[:, idx], eigenvals[idx]
                
                # Keep top eigenvectors
                d = min(len(eigenvals), self.memory_size)
                self.principal_dirs[name] = eigenvecs[:, :d]
                self.eigenvalues[name] = eigenvals[:d]
                
                # Store for visualization
                top_vals = eigenvals[:min(3, len(eigenvals))].detach().numpy()
                self._visualization_data['eigenvalues'][name].append(top_vals)
                
            except Exception as e:
                logger.warning(f"Topology update failed for {name}: {e}")

    def _update_topology_async(self) -> None:
        """Update topology asynchronously to avoid blocking training."""
        if self._topology_update_future and not self._topology_update_future.done():
            return  # Previous update still running
            
        self._topology_update_future = self.executor.submit(self._update_topology)

    def step(self, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Perform one TALT optimization step.
        
        Args:
            loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
            x: Input batch
            y: Target batch
            
        Returns:
            tuple: (loss_value, model_output)
        """
        self.optimizer.zero_grad()
        self.model.train()
        
        # Forward pass with mixed precision
        with autocast('cuda'):
            output = self.model(x)
            loss = loss_fn(output, y)
        
        # Backward pass (gradients computed and transformed by hooks)
        self.scaler.scale(loss).backward()
        
        # Store gradients periodically for eigenspace analysis
        if self.steps % self.store_interval == 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.grad_memory:
                    # Store flattened gradient on CPU
                    grad_flat = param.grad.detach().view(-1).clone().cpu()
                    self.grad_memory[name].append(grad_flat)
        
        # Parameter update using transformed gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update state
        self.steps += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self._visualization_data['loss_values'].append(loss_val)
        
        # Periodic topology update
        if self.steps % self.update_interval == 0:
            self._update_topology_async()
            
        return loss_val, output
    
    def step_complex(self, loss_fn: Callable, batch: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple], 
                    y: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """
        Complex step method for handling different input formats.
        
        Args:
            loss_fn: Loss function
            batch: Input batch - tensor for CNN or dict for transformers, or tuple/list of (x, y)
            y: Target tensor (optional for dict inputs)
            
        Returns:
            Tuple of (loss_value, model_output)
        """
        if isinstance(batch, dict):
            # Handle transformer/BERT inputs
            if y is None:
                y = batch.get('labels')
                if y is None:
                    raise ValueError("Dictionary input must contain 'labels' key or y must be provided")
            
            # Extract inputs and move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            token_type_ids = batch.get('token_type_ids', None)  
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            y = y.to(self.device)
            
            return self.step(loss_fn, input_ids, y)
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            # Handle standard tensor inputs - both tuple and list formats
            x, targets = batch
            x = x.to(self.device) if hasattr(x, 'to') else x
            targets = targets.to(self.device) if hasattr(targets, 'to') else targets
            return self.step(loss_fn, x, targets)
        else:
            # Handle single tensor case or unknown format
            if hasattr(batch, 'to'):
                batch = batch.to(self.device)
            if y is not None and hasattr(y, 'to'):
                y = y.to(self.device)
            return self.step(loss_fn, batch, y)

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients of the base optimizer for PyTorch compatibility.
        
        Args:
            set_to_none: Whether to set gradients to None instead of zero
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        """Access to underlying optimizer's parameter groups."""
        return self.optimizer.param_groups
    
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
        if self._topology_update_future and not self._topology_update_future.done():
            self._topology_update_future.cancel()

        # Shut down executor
        self.executor.shutdown(wait=False)

        # Clear memory
        self.grad_memory.clear()
        self._visualization_data.clear()

    def get_visualization_data(self):
        """Get visualization data for external analysis."""
        return {
            'loss_values': self._visualization_data['loss_values'],
            'bifurcation_points': self._visualization_data['bifurcation_points'],
            'eigenvalues': self._visualization_data['eigenvalues'],
            'bifurcations': self.bifurcations,
            'loss_history': self.loss_history
        }

    def _print_progress(self, loss_value: float, step: int) -> None:
        """Print training progress information."""
        if step % 10 == 0 or step == 1:
            print(f"Step {step}: Loss = {loss_value:.6f}")
