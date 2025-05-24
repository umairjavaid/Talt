"""
TensorBoard Logger for TALT Optimizer Visualization

This module provides comprehensive TensorBoard logging for TALT optimizers,
capturing both standard training metrics and TALT-specific metrics in real-time.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class TALTTensorBoardLogger:
    """
    Comprehensive TensorBoard logger for TALT optimizer experiments.
    
    Logs both standard training metrics and TALT-specific metrics including:
    - Loss and accuracy curves
    - Eigenvalue trajectories  
    - Valley detection events
    - Gradient norms and transformations
    - Parameter-wise curvature estimates
    - Comparative metrics between optimizers
    """
    
    def __init__(self, log_dir: str, experiment_name: str, 
                 flush_secs: int = 30, max_queue: int = 10):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of the experiment
            flush_secs: How often to flush logs to disk
            max_queue: Maximum number of outstanding logs
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")
        
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create experiment-specific log directory
        self.experiment_log_dir = self.log_dir / experiment_name
        self.experiment_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=str(self.experiment_log_dir),
            flush_secs=flush_secs,
            max_queue=max_queue
        )
        
        # Track logged metrics to avoid duplicates
        self.logged_metrics = set()
        
        # Store metric history for smoothing and analysis
        self.metric_history = defaultdict(list)
        
        # Track optimizer-specific loggers for comparison
        self.optimizer_writers = {}
        
        logger.info(f"Initialized TensorBoard logger for {experiment_name} in {self.experiment_log_dir}")
    
    def log_training_step(self, step: int, loss: float, accuracy: Optional[float] = None,
                         optimizer_state: Optional[Dict[str, Any]] = None,
                         optimizer_name: str = "main") -> None:
        """
        Log basic training metrics for a single step.
        
        Args:
            step: Training step number
            loss: Loss value
            accuracy: Accuracy value (optional)
            optimizer_state: Current optimizer state for TALT metrics
            optimizer_name: Name of optimizer for comparison logging
        """
        # Log basic metrics
        self.writer.add_scalar(f'Loss/Train/{optimizer_name}', loss, step)
        self.metric_history[f'loss_{optimizer_name}'].append(loss)
        
        if accuracy is not None:
            self.writer.add_scalar(f'Accuracy/Train/{optimizer_name}', accuracy, step)
            self.metric_history[f'accuracy_{optimizer_name}'].append(accuracy)
        
        # Log TALT-specific metrics if optimizer state is available
        if optimizer_state is not None:
            self.log_talt_metrics(step, optimizer_state, optimizer_name)
    
    def log_talt_metrics(self, step: int, optimizer_state: Dict[str, Any], 
                        optimizer_name: str = "talt") -> None:
        """
        Log TALT-specific metrics from optimizer state.
        
        Args:
            step: Training step number
            optimizer_state: TALT optimizer state containing internal metrics
            optimizer_name: Name prefix for metrics
        """
        try:
            # Extract TALT optimizer object if available
            talt_optimizer = None
            if hasattr(optimizer_state, 'get_tensorboard_metrics'):
                # Use new method if available
                metrics = optimizer_state.get_tensorboard_metrics()
                self._log_talt_metrics_dict(step, metrics, optimizer_name)
            elif hasattr(optimizer_state, 'get_visualization_data'):
                # Fallback to existing method
                viz_data = optimizer_state.get_visualization_data()
                self._log_visualization_data(step, viz_data, optimizer_name)
            elif isinstance(optimizer_state, dict):
                # Direct dictionary of metrics
                self._log_talt_metrics_dict(step, optimizer_state, optimizer_name)
            else:
                # Try to extract from optimizer object directly
                self._log_from_optimizer_object(step, optimizer_state, optimizer_name)
                
        except Exception as e:
            logger.warning(f"Failed to log TALT metrics at step {step}: {e}")
    
    def _log_talt_metrics_dict(self, step: int, metrics: Dict[str, Any], 
                              optimizer_name: str) -> None:
        """Log TALT metrics from a dictionary."""
        
        # Log eigenvalue trajectories
        if 'eigenvalues' in metrics:
            self._log_eigenvalues(step, metrics['eigenvalues'], optimizer_name)
        
        # Log gradient norms
        if 'gradient_norms' in metrics:
            self._log_gradient_norms(step, metrics['gradient_norms'], optimizer_name)
        
        # Log valley detection events
        if 'valley_detections' in metrics:
            self._log_valley_detections(step, metrics['valley_detections'], optimizer_name)
        
        # Log bifurcation points
        if 'bifurcations' in metrics:
            self._log_bifurcations(step, metrics['bifurcations'], optimizer_name)
        
        # Log curvature estimates
        if 'curvature_estimates' in metrics:
            self._log_curvature_estimates(step, metrics['curvature_estimates'], optimizer_name)
        
        # Log gradient transformation metrics
        if 'gradient_transformations' in metrics:
            self._log_gradient_transformations(step, metrics['gradient_transformations'], optimizer_name)
    
    def _log_visualization_data(self, step: int, viz_data: Dict[str, Any], 
                               optimizer_name: str) -> None:
        """Log metrics from get_visualization_data() output."""
        
        # Log eigenvalue history
        if 'eigenvalues_history' in viz_data:
            eigenvalue_data = {}
            for param_name, data in viz_data['eigenvalues_history'].items():
                if 'eigenvalues' in data and len(data['eigenvalues']) > 0:
                    eigenvalue_data[param_name] = data['eigenvalues'][-1]  # Latest eigenvalues
            if eigenvalue_data:
                self._log_eigenvalues(step, eigenvalue_data, optimizer_name)
        
        # Log gradient norm history
        if 'gradient_norms_history' in viz_data:
            gradient_norms = {}
            for param_name, data in viz_data['gradient_norms_history'].items():
                if 'grad_norms' in data and len(data['grad_norms']) > 0:
                    gradient_norms[param_name] = data['grad_norms'][-1]  # Latest norm
            if gradient_norms:
                self._log_gradient_norms(step, gradient_norms, optimizer_name)
        
        # Log valley detections
        if 'valley_detections' in viz_data and viz_data['valley_detections']:
            recent_detections = [d for d in viz_data['valley_detections'] if d[0] == step]
            if recent_detections:
                self._log_valley_detections(step, recent_detections, optimizer_name)
    
    def _log_from_optimizer_object(self, step: int, optimizer, optimizer_name: str) -> None:
        """Extract and log metrics directly from optimizer object."""
        
        # Try to get loss value
        if hasattr(optimizer, 'loss_history') and optimizer.loss_history:
            current_loss = optimizer.loss_history[-1]
            self.writer.add_scalar(f'Loss/TALT_Internal/{optimizer_name}', current_loss, step)
        
        # Try to get eigenvalue data
        if hasattr(optimizer, 'param_data'):  # ImprovedTALT
            eigenvalue_data = {}
            gradient_norms = {}
            
            for param_name, param_info in optimizer.param_data.items():
                # Extract gradient norms
                if 'gradient_norm_history' in param_info and param_info['gradient_norm_history']:
                    gradient_norms[param_name] = param_info['gradient_norm_history'][-1]
                
                # Extract eigenvalues from recent gradient stats
                if hasattr(optimizer, '_visualization_data'):
                    viz_data = optimizer._visualization_data
                    if 'gradient_stats' in viz_data and param_name in viz_data['gradient_stats']:
                        stats = list(viz_data['gradient_stats'][param_name])
                        if stats and isinstance(stats[-1], dict) and 'eigenvalues' in stats[-1]:
                            eigenvalue_data[param_name] = stats[-1]['eigenvalues']
            
            if eigenvalue_data:
                self._log_eigenvalues(step, eigenvalue_data, optimizer_name)
            if gradient_norms:
                self._log_gradient_norms(step, gradient_norms, optimizer_name)
        
        elif hasattr(optimizer, 'eigenvalues'):  # OriginalTALT
            eigenvalue_data = {}
            for param_name, eigenvals in optimizer.eigenvalues.items():
                if eigenvals is not None:
                    eigenvalue_data[param_name] = eigenvals.detach().cpu().numpy()
            
            if eigenvalue_data:
                self._log_eigenvalues(step, eigenvalue_data, optimizer_name)
    
    def _log_eigenvalues(self, step: int, eigenvalue_data: Dict[str, Any], 
                        optimizer_name: str) -> None:
        """Log eigenvalue trajectories for tracked parameters."""
        for param_name, eigenvals in eigenvalue_data.items():
            if isinstance(eigenvals, torch.Tensor):
                eigenvals = eigenvals.detach().cpu().numpy()
            
            # Log top 3-5 eigenvalues
            if isinstance(eigenvals, (list, np.ndarray)) and len(eigenvals) > 0:
                eigenvals = np.array(eigenvals)
                top_k = min(5, len(eigenvals))
                
                for i in range(top_k):
                    self.writer.add_scalar(
                        f'Eigenvalues/{optimizer_name}/{param_name}/Top_{i+1}',
                        eigenvals[i], step
                    )
                
                # Log eigenvalue distribution as histogram
                self.writer.add_histogram(
                    f'Eigenvalues_Dist/{optimizer_name}/{param_name}',
                    eigenvals, step
                )
    
    def _log_gradient_norms(self, step: int, gradient_norms: Dict[str, float], 
                           optimizer_name: str) -> None:
        """Log gradient norms per parameter group."""
        for param_name, norm in gradient_norms.items():
            self.writer.add_scalar(
                f'Gradient_Norms/{optimizer_name}/{param_name}',
                norm, step
            )
        
        # Log overall gradient norm statistics
        if gradient_norms:
            norms = list(gradient_norms.values())
            self.writer.add_scalar(f'Gradient_Norms/{optimizer_name}/Mean', np.mean(norms), step)
            self.writer.add_scalar(f'Gradient_Norms/{optimizer_name}/Max', np.max(norms), step)
            self.writer.add_scalar(f'Gradient_Norms/{optimizer_name}/Min', np.min(norms), step)
            self.writer.add_histogram(f'Gradient_Norms_Dist/{optimizer_name}', np.array(norms), step)
    
    def _log_valley_detections(self, step: int, valley_detections: List[Any], 
                              optimizer_name: str) -> None:
        """Log valley detection events."""
        # Count detections at this step
        detection_count = len(valley_detections)
        self.writer.add_scalar(f'Valley_Detections/{optimizer_name}/Count', detection_count, step)
        
        # Log as events for easy visualization
        for i, detection in enumerate(valley_detections):
            if isinstance(detection, (tuple, list)) and len(detection) >= 2:
                param_name = detection[1] if len(detection) > 1 else f"param_{i}"
                self.writer.add_scalar(f'Valley_Detections/{optimizer_name}/{param_name}', 1.0, step)
    
    def _log_bifurcations(self, step: int, bifurcations: List[int], 
                         optimizer_name: str) -> None:
        """Log bifurcation points."""
        # Check if current step is a bifurcation
        if step in bifurcations:
            self.writer.add_scalar(f'Bifurcations/{optimizer_name}', 1.0, step)
        
        # Log cumulative bifurcation count
        bifurcations_up_to_step = sum(1 for b in bifurcations if b <= step)
        self.writer.add_scalar(f'Bifurcations/{optimizer_name}/Cumulative', bifurcations_up_to_step, step)
    
    def _log_curvature_estimates(self, step: int, curvature_data: Dict[str, float], 
                                optimizer_name: str) -> None:
        """Log parameter-wise curvature estimates."""
        for param_name, curvature in curvature_data.items():
            self.writer.add_scalar(
                f'Curvature/{optimizer_name}/{param_name}',
                curvature, step
            )
    
    def _log_gradient_transformations(self, step: int, transform_data: Dict[str, Dict[str, float]], 
                                     optimizer_name: str) -> None:
        """Log gradient transformation metrics (before/after norms, etc.)."""
        for param_name, metrics in transform_data.items():
            for metric_name, value in metrics.items():
                self.writer.add_scalar(
                    f'Gradient_Transforms/{optimizer_name}/{param_name}/{metric_name}',
                    value, step
                )
    
    def log_validation_metrics(self, epoch: int, val_loss: float, val_accuracy: Optional[float] = None,
                              optimizer_name: str = "main") -> None:
        """Log validation metrics."""
        self.writer.add_scalar(f'Loss/Validation/{optimizer_name}', val_loss, epoch)
        if val_accuracy is not None:
            self.writer.add_scalar(f'Accuracy/Validation/{optimizer_name}', val_accuracy, epoch)
    
    def log_test_metrics(self, test_loss: float, test_accuracy: Optional[float] = None,
                        optimizer_name: str = "main") -> None:
        """Log final test metrics."""
        self.writer.add_scalar(f'Loss/Test/{optimizer_name}', test_loss, 0)
        if test_accuracy is not None:
            self.writer.add_scalar(f'Accuracy/Test/{optimizer_name}', test_accuracy, 0)
    
    def log_comparison_metrics(self, step: int, optimizer_name: str, 
                              metrics: Dict[str, float]) -> None:
        """Log metrics with optimizer prefix for comparison."""
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'Comparison/{metric_name}/{optimizer_name}', value, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Log hyperparameters and final metrics for hyperparameter tuning."""
        try:
            # Convert values to supported types
            clean_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    clean_hparams[key] = value
                else:
                    clean_hparams[key] = str(value)
            
            self.writer.add_hparams(clean_hparams, metrics)
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_loss_landscape_smoothness(self, step: int, loss_history: List[float],
                                     window_size: int = 10, optimizer_name: str = "main") -> None:
        """Log loss landscape smoothness metrics."""
        if len(loss_history) >= window_size:
            recent_losses = loss_history[-window_size:]
            variance = np.var(recent_losses)
            self.writer.add_scalar(f'Loss_Landscape/{optimizer_name}/Smoothness', variance, step)
    
    def log_parameter_norms(self, step: int, model: torch.nn.Module, 
                           optimizer_name: str = "main") -> None:
        """Log parameter norm changes."""
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_norm = param.data.norm(2).item()
                self.writer.add_scalar(f'Parameter_Norms/{optimizer_name}/{name}', param_norm, step)
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        self.writer.add_scalar(f'Parameter_Norms/{optimizer_name}/Total', total_norm, step)
    
    def create_comparison_plots(self, optimizers_data: Dict[str, Dict[str, List[float]]]) -> None:
        """Create side-by-side comparison plots for multiple optimizers."""
        try:
            import matplotlib.pyplot as plt
            
            # Create comparison plots for loss curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            for optimizer_name, data in optimizers_data.items():
                if 'train_loss' in data:
                    ax1.plot(data['train_loss'], label=f'{optimizer_name} Train')
                if 'val_loss' in data:
                    ax1.plot(data['val_loss'], label=f'{optimizer_name} Val', linestyle='--')
                if 'train_acc' in data:
                    ax2.plot(data['train_acc'], label=f'{optimizer_name} Train')
                if 'val_acc' in data:
                    ax2.plot(data['val_acc'], label=f'{optimizer_name} Val', linestyle='--')
            
            ax1.set_title('Loss Comparison')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            ax2.set_title('Accuracy Comparison')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            self.writer.add_figure('Optimizer_Comparison', fig, 0)
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to create comparison plots: {e}")
    
    def log_convergence_metrics(self, step: int, loss_history: List[float],
                               accuracy_history: Optional[List[float]] = None,
                               optimizer_name: str = "main") -> None:
        """Log convergence efficiency metrics."""
        if len(loss_history) < 10:
            return
        
        # Calculate convergence speed (steps to reach certain thresholds)
        final_loss = loss_history[-1]
        targets = [0.9, 0.95, 0.99]  # 90%, 95%, 99% of final performance
        
        for target in targets:
            threshold = final_loss * (2 - target)  # Lower is better for loss
            steps_to_threshold = None
            
            for i, loss in enumerate(loss_history):
                if loss <= threshold:
                    steps_to_threshold = i
                    break
            
            if steps_to_threshold is not None:
                self.writer.add_scalar(
                    f'Convergence/{optimizer_name}/Steps_to_{int(target*100)}pct',
                    steps_to_threshold, step
                )
        
        # Log similar metrics for accuracy if available
        if accuracy_history and len(accuracy_history) >= 10:
            final_acc = accuracy_history[-1]
            for target in targets:
                threshold = final_acc * target
                steps_to_threshold = None
                
                for i, acc in enumerate(accuracy_history):
                    if acc >= threshold:
                        steps_to_threshold = i
                        break
                
                if steps_to_threshold is not None:
                    self.writer.add_scalar(
                        f'Convergence/{optimizer_name}/Acc_Steps_to_{int(target*100)}pct',
                        steps_to_threshold, step
                    )
    
    def flush(self) -> None:
        """Manually flush logs to disk."""
        self.writer.flush()
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
        logger.info(f"Closed TensorBoard logger for {self.experiment_name}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_tensorboard_logger(log_dir: str, experiment_name: str) -> Optional[TALTTensorBoardLogger]:
    """
    Factory function to create a TensorBoard logger with error handling.
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Name of the experiment
        
    Returns:
        TALTTensorBoardLogger instance or None if TensorBoard is not available
    """
    try:
        return TALTTensorBoardLogger(log_dir, experiment_name)
    except ImportError:
        logger.warning("TensorBoard not available, skipping TensorBoard logging")
        return None
    except Exception as e:
        logger.error(f"Failed to create TensorBoard logger: {e}")
        return None
