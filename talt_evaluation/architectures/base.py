#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

class BaseArchitecture(nn.Module):
    """Base class for all architectures used in the evaluation framework."""
    
    def __init__(self, name, model_type):
        """
        Initialize the base architecture.
        
        Args:
            name: Name of the specific model
            model_type: Type of the model ('cnn', 'llm')
        """
        super(BaseArchitecture, self).__init__()
        self.name = name
        self.model_type = model_type
    
    def get_optimizer_config(self):
        """
        Get architecture-specific default TALT hyperparameters.
        
        Returns:
            dict: Default hyperparameters for the TALT optimizer
        """
        # Default TALT optimizer configuration
        if self.model_type == 'cnn':
            return {
                'projection_dim': 64,
                'memory_size': 10,
                'update_interval': 100,
                'valley_strength': 0.1,
                'smoothing_factor': 0.9,
                'grad_store_interval': 10,
                'cov_decay': 0.99,
                'adaptive_reg': 1e-5
            }
        elif self.model_type == 'llm':
            return {
                'projection_dim': 64,
                'memory_size': 5,
                'update_interval': 50,
                'valley_strength': 0.05,
                'smoothing_factor': 0.95,
                'grad_store_interval': 5,
                'cov_decay': 0.99,
                'adaptive_reg': 1e-6
            }
    
    def get_hyperparameter_search_space(self):
        """
        Define the search range for hyperparameter tuning.
        
        Returns:
            dict: Hyperparameter search space for Optuna
        """
        if self.model_type == 'cnn':
            return {
                'projection_dim': {'type': 'int', 'low': 16, 'high': 128},
                'memory_size': {'type': 'int', 'low': 5, 'high': 20},
                'update_interval': {'type': 'int', 'low': 50, 'high': 200},
                'valley_strength': {'type': 'float', 'low': 0.01, 'high': 0.5},
                'smoothing_factor': {'type': 'float', 'low': 0.8, 'high': 0.99},
                'grad_store_interval': {'type': 'int', 'low': 5, 'high': 20},
                'cov_decay': {'type': 'float', 'low': 0.9, 'high': 0.999},
                'adaptive_reg': {'type': 'float', 'low': 1e-6, 'high': 1e-4, 'log': True}
            }
        elif self.model_type == 'llm':
            return {
                'projection_dim': {'type': 'int', 'low': 16, 'high': 128},
                'memory_size': {'type': 'int', 'low': 3, 'high': 10},
                'update_interval': {'type': 'int', 'low': 20, 'high': 100},
                'valley_strength': {'type': 'float', 'low': 0.01, 'high': 0.2},
                'smoothing_factor': {'type': 'float', 'low': 0.9, 'high': 0.99},
                'grad_store_interval': {'type': 'int', 'low': 3, 'high': 10},
                'cov_decay': {'type': 'float', 'low': 0.95, 'high': 0.999},
                'adaptive_reg': {'type': 'float', 'low': 1e-7, 'high': 1e-5, 'log': True}
            }
    
    def architecture_specific_visualization(self, data):
        """
        Generate architecture-specific visualizations.
        
        Args:
            data: Input data for visualization
            
        Returns:
            dict: Visualization data
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")