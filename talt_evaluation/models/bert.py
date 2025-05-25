#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from transformers import BertModel, BertConfig

class BERTModel(nn.Module):
    """BERT model implementation for GLUE tasks."""
    
    def __init__(self, model_variant='base', num_classes=2, pretrained=True):
        """
        Initialize a BERT model.
        
        Args:
            model_variant: BERT variant ('base', 'large')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(BERTModel, self).__init__()
        self.name = f"bert-{model_variant}"
        self.model_type = 'llm'
        self.model_variant = model_variant
        self.num_classes = num_classes
        
        model_name = f'bert-{model_variant}-uncased'
        
        if pretrained:
            self.bert = BertModel.from_pretrained(model_name)
        else:
            config = BertConfig.from_pretrained(model_name)
            self.bert = BertModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_optimizer_config(self):
        """Get optimizer-specific configurations for this architecture."""
        return {
            'llm': {
                'lr': 2e-5,
                'weight_decay': 0.01,
                'eps': 1e-8
            },
            'improved-talt': {
                'lr': 2e-5,
                'memory_size': 10,
                'update_interval': 50,
                'valley_strength': 0.1,
                'smoothing_factor': 0.5,
                'grad_store_interval': 10,
                'min_param_size': 100,
                'max_param_size': 10000000,
                'sparsity_threshold': 0.01
            },
            'original-talt': {
                'lr': 2e-5,
                'memory_size': 10,
                'update_interval': 50,
                'valley_strength': 0.1,
                'smoothing_factor': 0.5,
                'grad_store_interval': 10,
                'min_param_size': 20
            }
        }
    
    def get_hyperparameter_search_space(self):
        """Get hyperparameter search space for this architecture."""
        return {
            'lr': {'type': 'float', 'low': 1e-6, 'high': 1e-4, 'log': True},
            'memory_size': {'type': 'int', 'low': 5, 'high': 20},
            'update_interval': {'type': 'int', 'low': 20, 'high': 100},
            'valley_strength': {'type': 'float', 'low': 0.05, 'high': 0.3},
            'smoothing_factor': {'type': 'float', 'low': 0.3, 'high': 0.8},
            'grad_store_interval': {'type': 'int', 'low': 5, 'high': 20},
            'min_param_size': {'type': 'int', 'low': 50, 'high': 1000},
            'max_param_size': {'type': 'int', 'low': 1000000, 'high': 50000000},
            'sparsity_threshold': {'type': 'float', 'low': 0.001, 'high': 0.1}
        }

def get_bert(model_variant='base', dataset='glue-sst2', pretrained=True):
    """
    Create a BERT model for the specified dataset.
    
    Args:
        model_variant: BERT variant ('base', 'large')
        dataset: Name of the dataset this model will be used with
        pretrained: Whether to use pretrained weights
    
    Returns:
        tuple: (model, model_config)
    """
    num_classes = 2  # Default for binary classification
    if dataset.lower() == 'glue-sst2':
        num_classes = 2
    # Add other dataset-specific num_classes if needed
    
    model = BERTModel(model_variant, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'variant': model_variant,
        'num_classes': num_classes
    }
    
    return model, model_config
