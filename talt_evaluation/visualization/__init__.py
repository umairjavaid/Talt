#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .training_visualization import create_training_report, plot_learning_curves, plot_optimizer_comparison
from .attention_visualization import plot_bert_attention
from .feature_visualization import plot_cnn_feature_maps

import json
import os
from datetime import datetime

def create_training_report(results, output_path=None):
    """
    Create a simple training report from results
    
    Args:
        results (dict): Dictionary containing training results
        output_path (str, optional): Path to save the report. If None, a timestamped path will be generated.
        
    Returns:
        str: Path to the saved report
    """
    # Try to use advanced visualization if available
    try:
        from .advanced import create_advanced_training_report
        return create_advanced_training_report(results, output_path)
    except ImportError:
        # Fall back to basic JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "accuracy": results.get("test_accuracy", 0),
                "loss": results.get("test_loss", 0),
                "epochs": len(results.get("train_losses", [])),
            }
        }
        
        if output_path is None:
            output_path = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Basic training report saved to {output_path}")
        return output_path

__all__ = ["create_training_report"]