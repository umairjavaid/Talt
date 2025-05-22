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

from .training_visualization import create_training_report as create_training_viz_report
from .training_visualization import plot_learning_curves, plot_optimizer_comparison
from .attention_visualization import plot_bert_attention
from .feature_visualization import plot_cnn_feature_maps
from .comprehensive_analysis import CrossExperimentAnalyzer

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
    else:
        results = results_or_experiment
        experiment = None
    
    # Try to use advanced visualization if available
    try:
        # Check if we have access to the visualization package
        from talt_evaluation.visualization.advanced import create_advanced_training_report
        logger.info("Using advanced visualization capabilities")
        return create_advanced_training_report(results, output_path)
    except ImportError:
        # Fall back to basic visualization from training_visualization
        logger.info("Using basic visualization capabilities")
        if experiment is not None:
            return create_training_viz_report(experiment, output_path or experiment.output_dir)
        else:
            # Fall back to basic JSON report if no advanced visualization and no experiment object
            report = {
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "summary": {
                    "accuracy": results.get("test_accuracy", results.get("test_acc", 0)),
                    "loss": results.get("test_loss", 0),
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
    "CrossExperimentAnalyzer"
]