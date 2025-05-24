#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TALT Evaluation Visualization Module

This module provides visualization tools for the TALT evaluation framework,
distinct from the core TALT visualization package.
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path

from .training_visualization import create_training_report as create_training_viz_report
from .training_visualization import plot_learning_curves, plot_optimizer_comparison
from .attention_visualization import plot_bert_attention
from .feature_visualization import plot_cnn_feature_maps
from .comprehensive_analysis import CrossExperimentAnalyzer
from .adaptive_visualizer import AdaptiveVisualizer

# Import TensorBoard logger with fallback
try:
    from .tensorboard_logger import TALTTensorBoardLogger, create_tensorboard_logger
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TALTTensorBoardLogger = None
    create_tensorboard_logger = None
    TENSORBOARD_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

def create_training_report(results_or_experiment, output_path=None):
    """
    Create a training report from results or an experiment object
    
    Args:
        results_or_experiment: Dictionary containing training results or an Experiment object
        output_path (str, optional): Path to save the report. If None, a timestamped path will be generated.
        
    Returns:
        str: Path to the saved report
    """
    # Handle case where an Experiment object is passed
    if hasattr(results_or_experiment, 'results'):
        experiment = results_or_experiment
        results = experiment.results
        output_dir = output_path or experiment.output_dir
        
        # Use adaptive visualizer for comprehensive visualization
        visualizer = AdaptiveVisualizer(
            output_dir=output_dir,
            experiment_name=getattr(experiment, 'name', 'experiment')
        )
        
        # Prepare experiment data
        experiment_data = results.copy()
        if hasattr(experiment, 'optimizer_type'):
            experiment_data['optimizer_type'] = experiment.optimizer_type
        if hasattr(experiment, 'model_config'):
            experiment_data['model_config'] = experiment.model_config
        
        # Prepare additional data (optimizer object, etc.)
        additional_data = {}
        if hasattr(experiment, 'optimizer'):
            additional_data['optimizer'] = experiment.optimizer
        
        # Generate all appropriate visualizations
        generated_viz = visualizer.generate_all_visualizations(experiment_data, additional_data)
        
        # Return path to visualization report
        viz_summary = visualizer.get_visualization_summary()
        logger.info(f"Generated {viz_summary['total_visualizations']} visualizations across {len(viz_summary['by_category'])} categories")
        
        return str(Path(output_dir) / f"{getattr(experiment, 'name', 'experiment')}_visualization_report.html")
        
    else:
        results = results_or_experiment
        experiment = None
    
    # Fallback to basic visualization from training_visualization
    logger.info("Using basic visualization capabilities")
    if experiment is not None:
        return create_training_viz_report(experiment, output_path or experiment.output_dir)
    else:
        # Fall back to basic JSON report if no advanced visualization and no experiment object
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "test_accuracy": results.get("test_accuracy", results.get("test_acc", 0)),
                "test_loss": results.get("test_loss", 0),
                "epochs": len(results.get("train_losses", results.get("train_loss", []))),
            }
        }
        
        if output_path is None:
            output_path = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Basic training report saved to {output_path}")
        return output_path

# Define the public API
__all__ = [
    "create_training_report",
    "plot_learning_curves",
    "plot_optimizer_comparison",
    "plot_bert_attention",
    "plot_cnn_feature_maps",
    "CrossExperimentAnalyzer",
    "AdaptiveVisualizer",
    "TALTTensorBoardLogger",
    "create_tensorboard_logger"
]