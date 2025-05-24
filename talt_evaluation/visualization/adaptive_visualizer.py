#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adaptive Visualization Coordinator

Intelligently generates appropriate visualizations based on:
- Optimizer type (TALT vs standard optimizers)
- Experiment mode (single, batch, comparison)
- Available data sources
- Architecture type (CNN, transformer, etc.)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import traceback

# Import visualization modules with error handling
try:
    from talt.visualization.visualizer import TALTVisualizer
    TALT_VIZ_AVAILABLE = True
except ImportError:
    TALT_VIZ_AVAILABLE = False
    TALTVisualizer = None

from .training_visualization import plot_learning_curves, plot_optimizer_comparison
from .comprehensive_analysis import CrossExperimentAnalyzer
from .feature_visualization import plot_cnn_feature_maps
try:
    from .attention_visualization import plot_bert_attention
    ATTENTION_VIZ_AVAILABLE = True
except ImportError:
    ATTENTION_VIZ_AVAILABLE = False
    plot_bert_attention = None

# Configure logging
logger = logging.getLogger(__name__)

class AdaptiveVisualizer:
    """
    Adaptive visualization coordinator that generates appropriate visualizations
    based on optimizer type, experiment mode, and available data.
    """
    
    def __init__(self, output_dir: Union[str, Path], experiment_name: str = "experiment"):
        """
        Initialize the adaptive visualizer.
        
        Args:
            output_dir: Base output directory for visualizations
            experiment_name: Name of the experiment for file naming
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Track generated visualizations for report
        self.generated_visualizations = {
            'talt_specific': [],
            'standard': [],
            'comparison': [],
            'architecture_specific': [],
            'cross_experiment': []
        }
        
        # Initialize TALT visualizer if available
        self.talt_visualizer = None
        if TALT_VIZ_AVAILABLE:
            try:
                self.talt_visualizer = TALTVisualizer(
                    output_dir=str(self.viz_dir),
                    max_points=1000
                )
            except Exception as e:
                logger.warning(f"Failed to initialize TALT visualizer: {e}")
        
        logger.info(f"Initialized adaptive visualizer for {experiment_name}")
    
    def detect_experiment_context(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect experiment context from available data.
        
        Args:
            experiment_data: Dictionary containing experiment results and metadata
            
        Returns:
            Dictionary with detected context information
        """
        context = {
            'optimizer_type': 'unknown',
            'is_talt': False,
            'architecture_type': 'unknown',
            'has_comparison_data': False,
            'has_batch_data': False,
            'available_data_sources': []
        }
        
        # Detect optimizer type
        if 'optimizer_type' in experiment_data:
            context['optimizer_type'] = experiment_data['optimizer_type']
            context['is_talt'] = experiment_data['optimizer_type'].lower() in ['talt', 'improved_talt']
        elif 'optimizer' in experiment_data:
            context['optimizer_type'] = experiment_data['optimizer']
            context['is_talt'] = experiment_data['optimizer'].lower() in ['talt', 'improved_talt']
        
        # Detect architecture type
        if 'model_config' in experiment_data:
            model_config = experiment_data['model_config']
            if 'name' in model_config:
                arch_name = model_config['name'].lower()
                if 'bert' in arch_name or 'transformer' in arch_name:
                    context['architecture_type'] = 'transformer'
                elif any(cnn_type in arch_name for cnn_type in ['resnet', 'vgg', 'cnn', 'efficientnet']):
                    context['architecture_type'] = 'cnn'
        
        # Check for comparison data (multiple optimizers)
        if 'comparison_results' in experiment_data:
            context['has_comparison_data'] = True
        
        # Check for batch data
        if 'batch_results' in experiment_data or 'experiments' in experiment_data:
            context['has_batch_data'] = True
        
        # Detect available data sources
        data_sources = []
        if 'train_loss' in experiment_data or 'train_acc' in experiment_data:
            data_sources.append('training_metrics')
        if 'test_loss' in experiment_data or 'test_acc' in experiment_data:
            data_sources.append('test_metrics')
        if context['is_talt'] and self.talt_visualizer:
            data_sources.append('talt_specific')
        if 'feature_maps' in experiment_data:
            data_sources.append('feature_maps')
        if 'attention_maps' in experiment_data:
            data_sources.append('attention_maps')
        
        context['available_data_sources'] = data_sources
        
        logger.info(f"Detected context: {context}")
        return context
    
    def generate_all_visualizations(self, experiment_data: Dict[str, Any], 
                                  additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Generate all appropriate visualizations based on experiment context.
        
        Args:
            experiment_data: Primary experiment data
            additional_data: Additional data sources (optimizer objects, etc.)
            
        Returns:
            Dictionary mapping visualization categories to lists of generated file paths
        """
        context = self.detect_experiment_context(experiment_data)
        
        logger.info(f"Generating visualizations for {context['optimizer_type']} optimizer")
        
        # Generate standard visualizations (always)
        self._generate_standard_visualizations(experiment_data, context)
        
        # Generate TALT-specific visualizations if applicable
        if context['is_talt'] and 'talt_specific' in context['available_data_sources']:
            self._generate_talt_visualizations(experiment_data, additional_data, context)
        
        # Generate architecture-specific visualizations
        if context['architecture_type'] != 'unknown':
            self._generate_architecture_visualizations(experiment_data, context)
        
        # Generate comparison visualizations if applicable
        if context['has_comparison_data']:
            self._generate_comparison_visualizations(experiment_data, context)
        
        # Generate cross-experiment analysis if batch data
        if context['has_batch_data']:
            self._generate_cross_experiment_visualizations(experiment_data, context)
        
        # Generate comprehensive HTML report
        self._generate_html_report(experiment_data, context)
        
        return self.generated_visualizations
    
    def _generate_standard_visualizations(self, experiment_data: Dict[str, Any], 
                                        context: Dict[str, Any]) -> None:
        """Generate standard training visualizations available for all optimizers."""
        try:
            # Learning curves
            if 'training_metrics' in context['available_data_sources']:
                learning_curves_path = self.viz_dir / f"{self.experiment_name}_learning_curves.png"
                plot_learning_curves(experiment_data, str(learning_curves_path))
                self.generated_visualizations['standard'].append(str(learning_curves_path))
                logger.info(f"Generated learning curves: {learning_curves_path}")
            
            # Training progress visualization
            self._create_training_progress_plot(experiment_data)
            
        except Exception as e:
            logger.error(f"Error generating standard visualizations: {e}")
            logger.debug(traceback.format_exc())
    
    def _generate_talt_visualizations(self, experiment_data: Dict[str, Any],
                                    additional_data: Optional[Dict[str, Any]],
                                    context: Dict[str, Any]) -> None:
        """Generate TALT-specific visualizations."""
        if not self.talt_visualizer:
            logger.warning("TALT visualizer not available, skipping TALT-specific visualizations")
            return
        
        try:
            # Extract TALT optimizer data
            talt_data = self._extract_talt_data(experiment_data, additional_data)
            
            if talt_data:
                # Add data to TALT visualizer
                self.talt_visualizer.add_optimizer_data(talt_data)
                
                # Generate loss trajectory with valley detection
                loss_traj_filename = f"{self.experiment_name}_talt_loss_trajectory.png"
                self.talt_visualizer.visualize_loss_trajectory(
                    save_path=loss_traj_filename,
                    show=False
                )
                loss_traj_path = self.viz_dir / loss_traj_filename
                if loss_traj_path.exists():
                    self.generated_visualizations['talt_specific'].append(str(loss_traj_path))
                
                # Generate eigenvalue spectra
                eigen_filename = f"{self.experiment_name}_talt_eigenvalues.png"
                self.talt_visualizer.visualize_eigenvalue_spectra(
                    save_path=eigen_filename,
                    show=False
                )
                eigen_path = self.viz_dir / eigen_filename
                if eigen_path.exists():
                    self.generated_visualizations['talt_specific'].append(str(eigen_path))
                
                # Generate gradient transformations
                grad_trans_filename = f"{self.experiment_name}_talt_grad_transformations.png"
                self.talt_visualizer.visualize_gradient_transformations(
                    save_path=grad_trans_filename,
                    show=False
                )
                grad_trans_path = self.viz_dir / grad_trans_filename
                if grad_trans_path.exists():
                    self.generated_visualizations['talt_specific'].append(str(grad_trans_path))
                
                # Generate gradient norm history
                grad_norm_filename = f"{self.experiment_name}_talt_grad_norms.png"
                self.talt_visualizer.visualize_gradient_norm_history(
                    save_path=grad_norm_filename,
                    show=False
                )
                grad_norm_path = self.viz_dir / grad_norm_filename
                if grad_norm_path.exists():
                    self.generated_visualizations['talt_specific'].append(str(grad_norm_path))
                
                # Generate loss landscape with valleys
                landscape_filename = f"{self.experiment_name}_talt_loss_landscape.png"
                self.talt_visualizer.visualize_loss_landscape_with_valleys(
                    save_path=landscape_filename,
                    show=False
                )
                landscape_path = self.viz_dir / landscape_filename
                if landscape_path.exists():
                    self.generated_visualizations['talt_specific'].append(str(landscape_path))
                
                logger.info(f"Generated {len(self.generated_visualizations['talt_specific'])} TALT-specific visualizations")
            else:
                logger.warning("No TALT data available for visualization")
            
        except Exception as e:
            logger.error(f"Error generating TALT visualizations: {e}")
            logger.debug(traceback.format_exc())
    
    def _generate_architecture_visualizations(self, experiment_data: Dict[str, Any],
                                            context: Dict[str, Any]) -> None:
        """Generate architecture-specific visualizations."""
        try:
            if context['architecture_type'] == 'cnn':
                self._generate_cnn_visualizations(experiment_data)
            elif context['architecture_type'] == 'transformer':
                self._generate_transformer_visualizations(experiment_data)
                
        except Exception as e:
            logger.error(f"Error generating architecture visualizations: {e}")
            logger.debug(traceback.format_exc())
    
    def _generate_cnn_visualizations(self, experiment_data: Dict[str, Any]) -> None:
        """Generate CNN-specific visualizations."""
        try:
            if 'feature_maps' in experiment_data:
                feature_maps_path = self.viz_dir / f"{self.experiment_name}_feature_maps"
                plot_cnn_feature_maps(
                    experiment_data['feature_maps'],
                    str(feature_maps_path)
                )
                # Add all generated feature map files
                for file_path in self.viz_dir.glob(f"{self.experiment_name}_feature_maps*"):
                    self.generated_visualizations['architecture_specific'].append(str(file_path))
                
                logger.info("Generated CNN feature map visualizations")
            
        except Exception as e:
            logger.error(f"Error generating CNN visualizations: {e}")
            logger.debug(traceback.format_exc())
    
    def _generate_transformer_visualizations(self, experiment_data: Dict[str, Any]) -> None:
        """Generate transformer-specific visualizations."""
        try:
            if ATTENTION_VIZ_AVAILABLE and 'attention_maps' in experiment_data:
                attention_path = self.viz_dir / f"{self.experiment_name}_attention_maps.png"
                plot_bert_attention(
                    experiment_data['attention_maps'],
                    str(attention_path)
                )
                self.generated_visualizations['architecture_specific'].append(str(attention_path))
                logger.info("Generated transformer attention visualizations")
            
        except Exception as e:
            logger.error(f"Error generating transformer visualizations: {e}")
            logger.debug(traceback.format_exc())
    
    def _generate_comparison_visualizations(self, experiment_data: Dict[str, Any],
                                          context: Dict[str, Any]) -> None:
        """Generate optimizer comparison visualizations."""
        try:
            if 'comparison_results' in experiment_data:
                comparison_data = experiment_data['comparison_results']
                
                # Extract standard and TALT results
                std_results = {}
                talt_results = {}
                
                for optimizer, results in comparison_data.items():
                    if optimizer.lower() in ['talt', 'improved_talt']:
                        talt_results = results
                    else:
                        std_results[optimizer] = results
                
                # Generate comparison plot
                if std_results and talt_results:
                    comparison_path = self.viz_dir / f"{self.experiment_name}_optimizer_comparison.png"
                    
                    # Use the first standard optimizer for comparison
                    first_std = next(iter(std_results.values()))
                    plot_optimizer_comparison(
                        first_std, talt_results,
                        save_path=str(comparison_path),
                        show=False
                    )
                    self.generated_visualizations['comparison'].append(str(comparison_path))
                    logger.info("Generated optimizer comparison visualization")
            
        except Exception as e:
            logger.error(f"Error generating comparison visualizations: {e}")
            logger.debug(traceback.format_exc())
    
    def _generate_cross_experiment_visualizations(self, experiment_data: Dict[str, Any],
                                                context: Dict[str, Any]) -> None:
        """Generate cross-experiment analysis visualizations."""
        try:
            # Use comprehensive analysis for batch experiments
            analyzer = CrossExperimentAnalyzer(self.output_dir.parent)
            
            # Generate performance matrix
            perf_matrix_path = self.viz_dir / f"{self.experiment_name}_performance_matrix.png"
            analyzer.generate_optimizer_performance_matrix(output_file=perf_matrix_path)
            self.generated_visualizations['cross_experiment'].append(str(perf_matrix_path))
            
            # Generate convergence analysis
            conv_analysis_dir = self.viz_dir / "convergence_analysis"
            analyzer.analyze_convergence_patterns(output_dir=conv_analysis_dir)
            
            # Add convergence plots to generated visualizations
            for conv_file in conv_analysis_dir.glob("*.png"):
                self.generated_visualizations['cross_experiment'].append(str(conv_file))
            
            # Generate efficiency analysis
            efficiency_path = self.viz_dir / f"{self.experiment_name}_efficiency_analysis.png"
            analyzer.generate_efficiency_analysis(output_file=efficiency_path)
            self.generated_visualizations['cross_experiment'].append(str(efficiency_path))
            
            logger.info("Generated cross-experiment analysis visualizations")
            
        except Exception as e:
            logger.error(f"Error generating cross-experiment visualizations: {e}")
            logger.debug(traceback.format_exc())
    
    def _create_training_progress_plot(self, experiment_data: Dict[str, Any]) -> None:
        """Create a comprehensive training progress plot."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Training Progress: {self.experiment_name}", fontsize=16)
            
            # Plot train/val loss
            if 'train_loss' in experiment_data and 'val_loss' in experiment_data:
                axes[0, 0].plot(experiment_data['train_loss'], label='Train Loss')
                axes[0, 0].plot(experiment_data['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('Loss Curves')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot train/val accuracy
            if 'train_acc' in experiment_data and 'val_acc' in experiment_data:
                axes[0, 1].plot(experiment_data['train_acc'], label='Train Accuracy')
                axes[0, 1].plot(experiment_data['val_acc'], label='Validation Accuracy')
                axes[0, 1].set_title('Accuracy Curves')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot learning rate schedule if available
            if 'learning_rates' in experiment_data:
                axes[1, 0].plot(experiment_data['learning_rates'])
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot training time per epoch if available
            if 'epoch_times' in experiment_data:
                axes[1, 1].plot(experiment_data['epoch_times'])
                axes[1, 1].set_title('Training Time per Epoch')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Time (seconds)')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            progress_path = self.viz_dir / f"{self.experiment_name}_training_progress.png"
            plt.savefig(progress_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.generated_visualizations['standard'].append(str(progress_path))
            logger.info(f"Generated training progress plot: {progress_path}")
            
        except Exception as e:
            logger.error(f"Error creating training progress plot: {e}")
            logger.debug(traceback.format_exc())
    
    def _extract_talt_data(self, experiment_data: Dict[str, Any],
                          additional_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract TALT-specific data from experiment results."""
        talt_data = {}
        
        try:
            # First, try to get optimizer from additional_data
            optimizer = None
            if additional_data:
                optimizer = additional_data.get('optimizer')
                # Also check for 'talt_optimizer' key
                if optimizer is None:
                    optimizer = additional_data.get('talt_optimizer')
            
            # Determine optimizer type for appropriate visualization
            optimizer_type = experiment_data.get('optimizer_type', '')
            
            # Run diagnostics if available
            if optimizer and hasattr(optimizer, 'diagnose_visualization_state'):
                logger.info("Running TALT diagnostics...")
                optimizer.diagnose_visualization_state()
            
            # Force topology update if we have very little data
            if optimizer and hasattr(optimizer, 'force_topology_update'):
                viz_data = optimizer.get_visualization_data() if hasattr(optimizer, 'get_visualization_data') else {}
                total_data_points = sum(len(v) if hasattr(v, '__len__') else 0 for v in viz_data.values())
                
                if total_data_points < 10:  # Very little data collected
                    logger.info("Insufficient TALT data detected, forcing topology update...")
                    optimizer.force_topology_update()
            
            # If we have an ImprovedTALTOptimizer instance, extract its data
            if optimizer and hasattr(optimizer, '_visualization_data'):
                logger.info("Extracting data from TALT optimizer instance")
                
                # Use the optimizer's get_visualization_data method if available
                if hasattr(optimizer, 'get_visualization_data'):
                    try:
                        opt_viz_data = optimizer.get_visualization_data()
                        # Start with the complete data from get_visualization_data
                        talt_data.update(opt_viz_data)
                        logger.info(f"Successfully extracted data using get_visualization_data: {list(opt_viz_data.keys())}")
                        
                        # Convert eigenvalues_history to TALTVisualizer format
                        if 'eigenvalues_history' in opt_viz_data and opt_viz_data['eigenvalues_history']:
                            eigenvalues_formatted = {}
                            for param_name, data in opt_viz_data['eigenvalues_history'].items():
                                if 'eigenvalues' in data and 'steps' in data:
                                    eigenvals_list = data['eigenvalues']
                                    steps_list = data['steps']
                                    # Convert to [(step, eigenvalues), ...] format
                                    eigenvalues_formatted[param_name] = [
                                        (step, eigenvals) for step, eigenvals in zip(steps_list, eigenvals_list)
                                    ]
                            
                            if eigenvalues_formatted:
                                talt_data['eigenvalues'] = eigenvalues_formatted
                                logger.info(f"Formatted eigenvalues data for {len(eigenvalues_formatted)} parameters")
                        
                        # Convert gradient_stats to grad_memory format for TALTVisualizer
                        if 'gradient_stats' in opt_viz_data and opt_viz_data['gradient_stats']:
                            grad_memory_formatted = {}
                            for param_name, stats_list in opt_viz_data['gradient_stats'].items():
                                if isinstance(stats_list, list) and len(stats_list) > 0:
                                    # Convert to [(step, grad_norm, 0), ...] format
                                    grad_memory_formatted[param_name] = []
                                    for stat_entry in stats_list:
                                        if isinstance(stat_entry, dict):
                                            step = stat_entry.get('step', 0)
                                            grad_norm = stat_entry.get('grad_norm', 0.0)
                                            grad_memory_formatted[param_name].append((step, grad_norm, 0))
                            
                            if grad_memory_formatted:
                                talt_data['grad_memory'] = grad_memory_formatted
                                logger.info(f"Formatted grad_memory data for {len(grad_memory_formatted)} parameters")
                        
                        # Alternative: Convert gradient_norms_history to grad_memory format
                        elif 'gradient_norms_history' in opt_viz_data and opt_viz_data['gradient_norms_history']:
                            grad_memory_formatted = {}
                            for param_name, data in opt_viz_data['gradient_norms_history'].items():
                                if 'grad_norms' in data:
                                    grad_norms = data['grad_norms']
                                    steps = data.get('steps', list(range(len(grad_norms))))
                                    # Convert to [(step, grad_norm, 0), ...] format
                                    grad_memory_formatted[param_name] = [
                                        (step, grad_norm, 0) for step, grad_norm in zip(steps, grad_norms)
                                    ]
                            
                            if grad_memory_formatted:
                                talt_data['grad_memory'] = grad_memory_formatted
                                logger.info(f"Formatted grad_memory from gradient_norms_history for {len(grad_memory_formatted)} parameters")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get visualization data from optimizer method: {e}")
                
                # Extract loss values if not already present
                if 'loss_values' not in talt_data or not talt_data['loss_values']:
                    if hasattr(optimizer, 'loss_history') and len(optimizer.loss_history) > 0:
                        talt_data['loss_history'] = list(optimizer.loss_history)
                        talt_data['loss_values'] = list(optimizer.loss_history)
                
                # Extract bifurcation points if not already present
                if 'bifurcations' not in talt_data or not talt_data['bifurcations']:
                    if hasattr(optimizer, 'bifurcations') and len(optimizer.bifurcations) > 0:
                        talt_data['bifurcations'] = list(optimizer.bifurcations)
                
                # Extract valley detections from visualization data if not already present
                if 'valley_detections' not in talt_data or not talt_data['valley_detections']:
                    viz_data = optimizer._visualization_data
                    if 'valley_detections' in viz_data and len(viz_data['valley_detections']) > 0:
                        talt_data['valley_detections'] = list(viz_data['valley_detections'])
        
            # Fallback: Extract from experiment data directly
            if not talt_data:
                logger.info("Falling back to experiment_data extraction")
                
                # Extract loss values
                if 'train_loss' in experiment_data:
                    talt_data['loss_values'] = experiment_data['train_loss']
                elif 'loss_history' in experiment_data:
                    talt_data['loss_history'] = experiment_data['loss_history']
                
                # Extract from experiment data directly
                if 'talt_visualization_data' in experiment_data:
                    talt_data.update(experiment_data['talt_visualization_data'])
            
            # Log what we extracted
            if talt_data:
                logger.info(f"Successfully extracted TALT data with keys: {list(talt_data.keys())}")
                for key, value in talt_data.items():
                    if isinstance(value, (list, dict)):
                        length = len(value) if hasattr(value, '__len__') else 'N/A'
                        logger.info(f"  {key}: {length} items")
                        # Log specific contents for key data structures
                        if key == 'eigenvalues' and isinstance(value, dict):
                            for param_name, param_data in value.items():
                                logger.info(f"    {param_name}: {len(param_data) if hasattr(param_data, '__len__') else 'N/A'} eigenvalue snapshots")
                        elif key == 'grad_memory' and isinstance(value, dict):
                            for param_name, param_data in value.items():
                                logger.info(f"    {param_name}: {len(param_data) if hasattr(param_data, '__len__') else 'N/A'} gradient history entries")
                    
                # Log specific TALTVisualizer format data
                if 'eigenvalues' in talt_data:
                    logger.info(f"  eigenvalues (TALTVisualizer format): {len(talt_data['eigenvalues'])} parameters")
                if 'grad_memory' in talt_data:
                    logger.info(f"  grad_memory (TALTVisualizer format): {len(talt_data['grad_memory'])} parameters")
            else:
                logger.warning("No TALT visualization data could be extracted")
            
            return talt_data if talt_data else None
        
        except Exception as e:
            logger.error(f"Error extracting TALT data: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _generate_html_report(self, experiment_data: Dict[str, Any], 
                            context: Dict[str, Any]) -> None:
        """Generate comprehensive HTML report with all visualizations."""
        try:
            report_path = self.output_dir / f"{self.experiment_name}_visualization_report.html"
            
            # Create HTML content
            html_content = self._create_html_content(experiment_data, context)
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated comprehensive HTML report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            logger.debug(traceback.format_exc())
    
    def _create_html_content(self, experiment_data: Dict[str, Any], 
                           context: Dict[str, Any]) -> str:
        """Create HTML content for the visualization report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visualization Report: {self.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin: 30px 0; }}
                .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                .viz-item {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .viz-item img {{ max-width: 100%; height: auto; }}
                .context-info {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .no-viz {{ color: #666; font-style: italic; }}
            </style>
        </head>
        <body>
            <h1>Visualization Report: {self.experiment_name}</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="context-info">
                <h3>Experiment Context</h3>
                <p><strong>Optimizer:</strong> {context['optimizer_type']}</p>
                <p><strong>Architecture:</strong> {context['architecture_type']}</p>
                <p><strong>TALT Optimizer:</strong> {'Yes' if context['is_talt'] else 'No'}</p>
                <p><strong>Available Data Sources:</strong> {', '.join(context['available_data_sources'])}</p>
            </div>
        """
        
        # Add standard visualizations section
        html += self._add_visualization_section("Standard Training Visualizations", 
                                               self.generated_visualizations['standard'])
        
        # Add TALT-specific visualizations if available
        if context['is_talt'] and self.generated_visualizations['talt_specific']:
            html += self._add_visualization_section("TALT-Specific Visualizations", 
                                                   self.generated_visualizations['talt_specific'])
        
        # Add architecture-specific visualizations
        if self.generated_visualizations['architecture_specific']:
            html += self._add_visualization_section("Architecture-Specific Visualizations", 
                                                   self.generated_visualizations['architecture_specific'])
        
        # Add comparison visualizations
        if self.generated_visualizations['comparison']:
            html += self._add_visualization_section("Optimizer Comparison Visualizations", 
                                                   self.generated_visualizations['comparison'])
        
        # Add cross-experiment visualizations
        if self.generated_visualizations['cross_experiment']:
            html += self._add_visualization_section("Cross-Experiment Analysis", 
                                                   self.generated_visualizations['cross_experiment'])
        
        html += """
            </body>
        </html>
        """
        
        return html
    
    def _add_visualization_section(self, title: str, viz_paths: List[str]) -> str:
        """Add a section of visualizations to the HTML report."""
        if not viz_paths:
            return f"""
            <div class="section">
                <h2>{title}</h2>
                <p class="no-viz">No visualizations available for this category.</p>
            </div>
            """
        
        html = f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="viz-grid">
        """
        
        for viz_path in viz_paths:
            viz_file = Path(viz_path)
            if viz_file.exists():
                rel_path = viz_file.relative_to(self.output_dir)
                html += f"""
                <div class="viz-item">
                    <h3>{viz_file.stem.replace('_', ' ').title()}</h3>
                    <img src="{rel_path}" alt="{viz_file.stem}">
                </div>
                """
        
        html += """
            </div>
        </div>
        """
        
        return html

    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get a summary of all generated visualizations."""
        total_visualizations = sum(len(viz_list) for viz_list in self.generated_visualizations.values())
        
        return {
            'total_visualizations': total_visualizations,
            'by_category': {cat: len(viz_list) for cat, viz_list in self.generated_visualizations.items()},
            'visualization_paths': self.generated_visualizations
        }
