#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from huggingface_hub.utils import disable_progress_bars
# Import from Hugging Face datasets library explicitly
from huggingface_hub import HfApi
import datasets as hf_datasets
from transformers import BertTokenizer

# Disable progress bars for cleaner output
disable_progress_bars()

class GlueDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def get_glue_dataset(task_name, batch_size=32, num_workers=4):
    """
    Get a GLUE dataset preprocessed for BERT fine-tuning.
    Currently supports SST-2 dataset.
    
    Args:
        task_name: GLUE task name ('sst2' currently supported)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if task_name.lower() != 'sst2':
        raise ValueError("Currently only 'sst2' GLUE task is supported")
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128  # Maximum sequence length
    
    # Load SST-2 dataset from Hugging Face datasets
    dataset = hf_datasets.load_dataset("glue", "sst2")
    
    # Preprocess training data
    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding='max_length',
        max_length=max_length
    )
    
    # Preprocess validation data
    val_texts = dataset["validation"]["sentence"]
    val_labels = dataset["validation"]["label"]
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding='max_length',
        max_length=max_length
    )
    
    # Preprocess test data - normally there's no test set with labels available
    # for GLUE benchmarks, but here we'll use the validation set as a test set for
    # demonstration purposes
    test_texts = dataset["validation"]["sentence"]  # Using validation as test
    test_labels = dataset["validation"]["label"]    # Using validation as test
    test_encodings = tokenizer(
        test_texts, 
        truncation=True, 
        padding='max_length',
        max_length=max_length
    )
    
    # Create dataset objects
    train_dataset = GlueDataset(train_encodings, train_labels)
    val_dataset = GlueDataset(val_encodings, val_labels)
    test_dataset = GlueDataset(test_encodings, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader