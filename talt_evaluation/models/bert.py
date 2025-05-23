#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import BaseArchitecture
from transformers import BertModel as HFBertModel, BertConfig
import torch.nn as nn

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
            self.bert = HFBertModel.from_pretrained(model_name)
        else:
            config = BertConfig.from_pretrained(model_name)
            self.bert = HFBertModel(config)
        
        # Get the hidden size dimension based on variant
        self.hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize weights
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
        # Setup for attention visualization
        self.attention_maps = {}
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, store_attention=False):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=store_attention
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled_output))
        
        if store_attention:
            self.attention_maps = outputs.attentions
            
        return logits
    
    # get_optimizer_config and get_hyperparameter_search_space can be inherited or overridden
    # architecture_specific_visualization can be inherited or overridden

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
    num_classes = 2 # Default for glue-sst2
    if dataset.lower() == 'glue-mnli':
        num_classes = 3
    # Add other dataset-specific num_classes if needed
    
    model = BERTModel(variant, num_classes=num_classes, pretrained=pretrained)
    
    model_config = {
        'name': model.name,
        'model_type': model.model_type,
        'variant': variant,
        'num_classes': num_classes,
        'hidden_size': model.hidden_size
    }
    
    return model, model_config
