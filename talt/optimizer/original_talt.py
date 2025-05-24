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
        registered_params = 0
        total_params = 0
        
        # Make threshold adaptive for better coverage
        total_model_params = sum(p.numel() for p in model.parameters())
        adaptive_threshold = max(10, min(self.min_param_size, int(total_model_params * 0.001)))
        
        logger.info(f"Original TALT initialized: memory_size={eigenspace_memory_size}, update_interval={topology_update_interval}")
        logger.info(f"Model has {total_model_params} total parameters")
        logger.info(f"Using adaptive threshold: {adaptive_threshold} (original: {self.min_param_size})")
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.requires_grad and param.numel() > adaptive_threshold:
                registered_params += 1
                self.grad_memory[name] = deque(maxlen=self.memory_size)
                self.principal_dirs[name] = None
                self.eigenvalues[name] = None
                self._visualization_data['eigenvalues'][name] = []
                
                logger.debug(f"Registered Original TALT hook for {name} (size: {param.numel()})")
                
                # Hook transforms gradients using eigenspace analysis
                param.register_hook(lambda grad, name=name: self._transform_gradient(grad, name))
        
        logger.info(f"Original TALT: Registered hooks for {registered_params}/{total_params} parameters")
        if registered_params == 0:
            logger.warning("No parameters registered for Original TALT! Consider lowering min_param_size.")

    def _transform_gradient(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """
        Transform gradient using exact TALT algorithm.
        
        Args:
            grad: Original gradient tensor
            name: Parameter name
            
        Returns:
            Transformed gradient tensor
        """
        if name not in self.grad_memory:
            return grad
        
        # Log gradient transform calls periodically
        if self.steps % 50 == 0:
            logger.debug(f"Original TALT gradient transform called for {name} at step {self.steps}")
            
        try:
            # Store gradient in memory
            flat_grad = grad.view(-1).detach().cpu()
            self.grad_memory[name].append(flat_grad)
            
            # Only proceed if we have enough gradients
            if len(self.grad_memory[name]) < 3:
                return grad
                
            # Convert to matrix G = [g₁, g₂, ..., gₖ]
            grad_matrix = torch.stack(list(self.grad_memory[name]), dim=1)  # [d, k]
            
            # Center the gradients: Ḡ = G - mean(G)
            grad_mean = grad_matrix.mean(dim=1, keepdim=True)
            centered_grads = grad_matrix - grad_mean
            
            # Compute covariance matrix: C = ḠᵀḠ/(k-1)
            k = centered_grads.shape[1]
            covariance = torch.matmul(centered_grads.t(), centered_grads) / (k - 1)
            
            # Add regularization for numerical stability
            reg = 1e-6 * torch.eye(covariance.shape[0])
            covariance = covariance + reg
            
            # Eigendecomposition: C = VΛVᵀ
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                
                logger.debug(f"Original TALT computed {len(eigenvalues)} eigenvalues for {name}")
                
                # Store eigenvalues for visualization
                self.eigenvalues[name] = eigenvalues.detach()
                self.principal_dirs[name] = eigenvectors.detach()
                
                # Project current gradient into eigenspace
                current_centered = flat_grad - grad_mean.squeeze()
                coeffs = torch.matmul(eigenvectors.t(), current_centered.unsqueeze(1)).squeeze()
                
                # Transform coefficients based on eigenvalues
                # High eigenvalues (high curvature) -> reduce step size
                # Low eigenvalues (flat regions) -> increase step size
                eigenvals_clamped = torch.clamp(eigenvalues, min=1e-8)
                
                scale_factors = torch.ones_like(eigenvals_clamped)
                
                # For high curvature directions (large eigenvalues)
                high_curvature_mask = eigenvals_clamped > 1.0
                scale_factors[high_curvature_mask] = self.smoothing_factor / torch.sqrt(eigenvals_clamped[high_curvature_mask])
                
                # For flat regions (small eigenvalues)
                flat_mask = eigenvals_clamped < 0.1
                scale_factors[flat_mask] = 1.0 + self.valley_strength
                
                # Apply transformations
                transformed_coeffs = coeffs * scale_factors
                
                # Reconstruct gradient
                transformed_grad = torch.matmul(eigenvectors, transformed_coeffs.unsqueeze(1)).squeeze()
                
                # Add back the mean
                transformed_grad = transformed_grad + grad_mean.squeeze()
                
                # Store visualization data
                if len(self._visualization_data['eigenvalues'][name]) < 1000:  # Limit storage
                    self._visualization_data['eigenvalues'][name].append(
                        (self.steps, eigenvalues.detach().cpu().numpy())
                    )
                
                return transformed_grad.view_as(grad).to(grad.device)
                
            except RuntimeError as e:
                logger.warning(f"Eigendecomposition failed for {name}: {e}")
                return grad
                
        except Exception as e:
            logger.error(f"Error in gradient transformation for {name}: {e}")
            return grad

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
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            # Handle standard tensor inputs - both tuple and list formats
            x, targets = batch[0], batch[1]  # Extract first two elements
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
        logger.info("Collecting Original TALT visualization data...")
        
        # Log data availability
        for name, eig_data in self._visualization_data['eigenvalues'].items():
            logger.debug(f"{name}: {len(eig_data)} eigenvalue snapshots")
        
        viz_data = {
            'loss_values': self._visualization_data['loss_values'],
            'bifurcation_points': self._visualization_data['bifurcation_points'],
            'eigenvalues': self._visualization_data['eigenvalues'],
            'bifurcations': self.bifurcations,
            'loss_history': self.loss_history,
            'grad_memory': self.grad_memory
        }
        
        logger.info(f"Original TALT visualization data collected: "
                    f"{len(viz_data['eigenvalues'])} params with eigenvalues, "
                    f"{len(viz_data['bifurcations'])} bifurcations, "
                    f"{len(viz_data['loss_history'])} loss values")
        
        return viz_data

    def _print_progress(self, loss_value: float, step: int) -> None:
        """Print training progress information."""
        if step % 10 == 0 or step == 1:
            print(f"Step {step}: Loss = {loss_value:.6f}")
