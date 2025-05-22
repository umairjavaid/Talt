#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import time
import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Experiment:
    """
    Experiment class to handle training, evaluation, and result tracking.
    """
    
    def __init__(self, model, model_config, train_loader, val_loader, test_loader,
                 optimizer_type, optimizer_config, epochs, device, output_dir,
                 mixed_precision=False, save_checkpoints=False, checkpoint_interval=5):
        """
        Initialize an experiment.
        
        Args:
            model: Model to train
            model_config: Configuration of the model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer_type: Type of optimizer ('talt', 'sgd', 'adam')
            optimizer_config: Configuration for the optimizer
            epochs: Number of training epochs
            device: Device to train on
            output_dir: Directory to save results
            mixed_precision: Whether to use mixed precision training
            save_checkpoints: Whether to save checkpoints
            checkpoint_interval: Interval (in epochs) for saving checkpoints
        """
        self.model = model
        self.model_config = model_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config
        self.epochs = epochs
        self.device = device
        self.output_dir = output_dir
        self.mixed_precision = mixed_precision
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        
        # Create results directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model on device
        self.model = self.model.to(device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Set loss function based on model type
        if self.model.model_type == 'cnn':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:  # For BERT
            self.criterion = torch.nn.CrossEntropyLoss()
        
        # Use mixed precision for faster training
        self.scaler = GradScaler() if mixed_precision and device.type == 'cuda' else None
        
        # Track metrics
        self.results = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': None,
            'test_acc': None,
            'best_epoch': 0,
            'best_val_acc': 0.0,
            'training_time': 0.0,
            'model_config': model_config,
            'optimizer_type': optimizer_type,
            'optimizer_config': optimizer_config
        }
        
        # Initialize best model weights
        self.best_model_weights = None
    
    def _create_optimizer(self):
        """
        Create optimizer based on specified type and configuration.
        
        Returns:
            torch.optim.Optimizer: The initialized optimizer
        """
        if self.optimizer_type == 'talt':
            try:
                # Fixed: Use ImprovedTALTOptimizer instead of TALT
                from talt.optimizer import ImprovedTALTOptimizer as TALT
                optimizer = TALT(
                    self.model.parameters(),
                    **self.optimizer_config
                )
                logger.info("Created TALT optimizer")
            except ImportError:
                logger.error("TALT optimizer not available. Install TALT package first.")
                raise
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.optimizer_config.get('lr', 0.1),
                momentum=self.optimizer_config.get('momentum', 0.9),
                weight_decay=self.optimizer_config.get('weight_decay', 5e-4)
            )
            logger.info("Created SGD optimizer")
        elif self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optimizer_config.get('lr', 0.001),
                weight_decay=self.optimizer_config.get('weight_decay', 0)
            )
            logger.info("Created Adam optimizer")
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        return optimizer
    
    def _train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            tuple: (train_loss, train_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different dataset types
            if isinstance(batch, dict):  # For BERT/transformer models
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.mixed_precision and self.scaler is not None:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        )
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                _, predicted = torch.max(outputs, 1)
                
            else:  # For CNN models
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.mixed_precision and self.scaler is not None:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                _, predicted = torch.max(outputs, 1)
            
            total_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': correct / total * 100.0
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate(self):
        """
        Validate the model.
        
        Returns:
            tuple: (validation_loss, validation_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Handle different dataset types
                if isinstance(batch, dict):  # For BERT/transformer models
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch.get('token_type_ids')
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    
                else:  # For CNN models
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                total_loss += loss.item()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def test(self):
        """
        Test the model on the test set.
        
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        # Load best model weights if available
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Handle different dataset types
                if isinstance(batch, dict):  # For BERT/transformer models
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch.get('token_type_ids')
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    
                else:  # For CNN models
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                total_loss += loss.item()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        test_loss = total_loss / len(self.test_loader)
        test_acc = correct / total
        
        # Update results
        self.results['test_loss'] = test_loss
        self.results['test_acc'] = test_acc
        
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        return test_loss, test_acc
    
    def run(self):
        """
        Run the experiment (train and evaluate).
        
        Returns:
            dict: Experiment results
        """
        logger.info(f"Starting experiment with {self.optimizer_type} optimizer")
        logger.info(f"Model: {self.model_config['name']}")
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self._validate()
            
            # Update results
            self.results['train_loss'].append(train_loss)
            self.results['train_acc'].append(train_acc)
            self.results['val_loss'].append(val_loss)
            self.results['val_acc'].append(val_acc)
            
            # Check if this is the best model
            if val_acc > self.results['best_val_acc']:
                self.results['best_val_acc'] = val_acc
                self.results['best_epoch'] = epoch
                self.best_model_weights = self.model.state_dict().copy()
                
                # Save best model
                if self.save_checkpoints:
                    self._save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if self.save_checkpoints and (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Calculate training time
        self.results['training_time'] = time.time() - start_time
        
        # Test model
        self.test()
        
        # Save results
        self._save_results()
        
        logger.info(f"Experiment completed in {self.results['training_time']:.2f} seconds")
        logger.info(f"Best validation accuracy: {self.results['best_val_acc']:.4f} "
                    f"at epoch {self.results['best_epoch']+1}")
        logger.info(f"Test accuracy: {self.results['test_acc']:.4f}")
        
        return self.results
    
    def _save_checkpoint(self, epoch, is_best=False):
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filename = f"checkpoint_epoch_{epoch+1}.pt"
        if is_best:
            filename = "best_model.pt"
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'results': self.results
        }
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_results(self):
        """Save experiment results to disk."""
        results_path = os.path.join(self.output_dir, 'results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load results history
        self.results = checkpoint['results']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {checkpoint['epoch']+1}")
        
        return checkpoint['epoch']