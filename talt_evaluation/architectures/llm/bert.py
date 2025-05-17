#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from ..base import BaseArchitecture

class BERTModel(BaseArchitecture):
    """BERT model implementation for text classification tasks."""
    
    def __init__(self, variant='base', num_classes=2, pretrained=True):
        """
        Initialize a BERT model.
        
        Args:
            variant: BERT variant ('base', 'large')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(BERTModel, self).__init__(f"bert-{variant}", 'llm')
        self.variant = variant
        self.num_classes = num_classes
        
        # Initialize the BERT model
        model_name = f"bert-{variant}-uncased"
        if pretrained:
            self.bert = BertModel.from_pretrained(model_name)
        else:
            config = BertConfig.from_pretrained(model_name)
            self.bert = BertModel(config)
        
        # Get the hidden size dimension based on variant
        if variant == 'base':
            self.hidden_size = 768
        elif variant == 'large':
            self.hidden_size = 1024
        else:
            raise ValueError(f"Unsupported BERT variant: {variant}")
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize weights
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
        # Setup for attention visualization
        self.attention_maps = {}
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, store_attention=False):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            store_attention: Whether to store attention maps for visualization
        
        Returns:
            torch.Tensor: Logits for each class
        """
        # Use output_attentions=True if we need to visualize attention
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=store_attention
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Store attention maps if requested
        if store_attention:
            self.attention_maps = {
                f"layer_{i}": attention.detach().cpu()
                for i, attention in enumerate(outputs.attentions)
            }
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_optimizer_config(self):
        """Get BERT-specific default TALT hyperparameters."""
        config = super().get_optimizer_config()
        
        # BERT-specific adjustments based on the variant
        if self.variant == 'base':
            config.update({
                'projection_dim': 64,
                'memory_size': 5,
                'update_interval': 50,
                'valley_strength': 0.05,
                'smoothing_factor': 0.95,
                'grad_store_interval': 5
            })
        elif self.variant == 'large':
            config.update({
                'projection_dim': 128,
                'memory_size': 3,
                'update_interval': 30,
                'valley_strength': 0.03,
                'smoothing_factor': 0.97,
                'grad_store_interval': 3
            })
        
        return config
    
    def get_hyperparameter_search_space(self):
        """Define search space for hyperparameter tuning."""
        search_space = super().get_hyperparameter_search_space()
        
        # BERT-specific search space adjustments
        if self.variant == 'large':
            search_space.update({
                'projection_dim': {'type': 'int', 'low': 64, 'high': 256},
                'valley_strength': {'type': 'float', 'low': 0.01, 'high': 0.1}
            })
        
        return search_space
    
    def architecture_specific_visualization(self, data):
        """
        Generate BERT-specific visualizations, particularly attention maps.
        
        Args:
            data: Input data batch with 'input_ids', 'attention_mask', etc.
            
        Returns:
            dict: Visualization data
        """
        self.bert.eval()
        
        # Run a forward pass with attention output enabled
        with torch.no_grad():
            _ = self.forward(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                token_type_ids=data.get('token_type_ids'),
                store_attention=True
            )
        
        # Prepare visualization data
        visualizations = {
            'model_type': self.model_type,
            'name': self.name,
            'attention_maps': self.attention_maps
        }
        
        return visualizations


def get_bert(variant, dataset='glue-sst2', pretrained=True):
    """
    Create a BERT model for the specified dataset.
    
    Args:
        variant: BERT variant ('base', 'large')
        dataset: Name of the dataset this model will be used with
        pretrained: Whether to use pretrained weights
    
    Returns:
        tuple: (model, model_config)
    """
    if dataset.lower() == 'glue-sst2':
        num_classes = 2  # Binary sentiment classification
    else:
        raise ValueError(f"Unsupported dataset for BERT: {dataset}")
    
    model = BERTModel(variant, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'variant': variant,
        'num_classes': num_classes,
        'hidden_size': model.hidden_size
    }
    
    return model, model_config