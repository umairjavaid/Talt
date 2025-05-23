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
        else:
            return {} # Default empty config
    
    def get_hyperparameter_search_space(self):
        """
        Define the search range for hyperparameter tuning.
        
        Returns:
            dict: Hyperparameter search space for Optuna
        """
        if self.model_type == 'cnn':
            pass # To be implemented by subclasses or filled
        elif self.model_type == 'llm':
            pass # To be implemented by subclasses or filled
        return {} # Default empty search space
    
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

