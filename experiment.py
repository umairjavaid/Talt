"""Experiment script for training and evaluating models with TALT."""

import os
import time
import torch
import logging
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load checkpoint with improved error handling and compatibility checks.

    Args:
        model: The model to load the state dict into.
        optimizer: The optimizer to load the state dict into.
        scheduler: The learning rate scheduler to load the state dict into.
        checkpoint_path: Path to the checkpoint file.
        device: Device to map the checkpoint tensors to.

    Returns:
        The epoch number the checkpoint was saved from, or 0 if loading failed.
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Check optimizer compatibility
        if 'optimizer_type' in checkpoint:
            current_type = type(optimizer).__name__
            saved_type = checkpoint['optimizer_type']
            if current_type != saved_type:
                logger.warning(f"Optimizer type mismatch: current={current_type}, saved={saved_type}")
                logger.warning("Skipping optimizer state loading due to incompatibility")
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Try to load optimizer state, but handle failures gracefully
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
                logger.warning("Continuing with fresh optimizer state")
        
        # Load scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_path):
    """
    Save checkpoint with optimizer type information.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The learning rate scheduler to save.
        epoch: The current epoch number.
        loss: The current loss value.
        checkpoint_path: Path to save the checkpoint file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_type': type(optimizer).__name__,
        'loss': loss
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")