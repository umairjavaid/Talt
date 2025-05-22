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
        
        # Use mixed precision for faster training - only for standard optimizers
        # TALT optimizer handles mixed precision internally
        self.scaler = GradScaler() if mixed_precision and device.type == 'cuda' and optimizer_type != 'talt' else None
        
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
                # Import TALT optimizer
                from talt.optimizer import ImprovedTALTOptimizer as TALT
                
                # Extract base optimizer parameters
                base_optimizer_params = {
                    'lr': self.optimizer_config.get('lr', 0.01),
                    'momentum': self.optimizer_config.get('momentum', 0.9),
                    'weight_decay': self.optimizer_config.get('weight_decay', 5e-4)
                }
                
                # Extract TALT-specific parameters
                talt_params = {
                    'lr': self.optimizer_config.get('lr', 0.01),
                    'projection_dim': self.optimizer_config.get('projection_dim', 32),
                    'memory_size': self.optimizer_config.get('memory_size', 10),
                    'update_interval': self.optimizer_config.get('update_interval', 20),
                    'valley_strength': self.optimizer_config.get('valley_strength', 0.2),
                    'smoothing_factor': self.optimizer_config.get('smoothing_factor', 0.3),
                    'grad_store_interval': self.optimizer_config.get('grad_store_interval', 5),
                    'cov_decay': self.optimizer_config.get('cov_decay', 0.95),
                    'adaptive_reg': self.optimizer_config.get('adaptive_reg', True),
                    'device': self.device
                }
                
                # Create base optimizer factory that properly receives the base parameters
                base_optimizer = lambda params, lr: torch.optim.SGD(
                    params, 
                    lr=lr,
                    momentum=base_optimizer_params['momentum'],
                    weight_decay=base_optimizer_params['weight_decay']
                )
                
                # Create TALT optimizer with appropriate parameters
                optimizer = TALT(
                    model=self.model,
                    base_optimizer=base_optimizer,
                    **talt_params
                )
                logger.info("Created TALT optimizer with properly separated parameters")
            except ImportError as e:
                logger.error(f"TALT optimizer not available: {e}")
                raise
            except Exception as e:
                logger.error(f"Error creating TALT optimizer: {e}")
                raise
        elif self.optimizer_type == 'sgd':
            try:
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.optimizer_config.get('lr', 0.1),
                    momentum=self.optimizer_config.get('momentum', 0.9),
                    weight_decay=self.optimizer_config.get('weight_decay', 5e-4)
                )
                logger.info("Created SGD optimizer")
            except Exception as e:
                logger.error(f"Error creating SGD optimizer: {e}")
                raise
        elif self.optimizer_type == 'adam':
            try:
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.optimizer_config.get('lr', 0.001),
                    weight_decay=self.optimizer_config.get('weight_decay', 0)
                )
                logger.info("Created Adam optimizer")
            except Exception as e:
                logger.error(f"Error creating Adam optimizer: {e}")
                raise
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        return optimizer
    
    def _train_epoch(self, epoch):
        """
        Train for one epoch with proper handling for different optimizer types.
        
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
            if self.optimizer_type == 'talt':
                # TALT optimizer has its own step method that handles forward/backward/optimize
                if isinstance(batch, dict):  # For BERT/transformer models
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch.get('token_type_ids')
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Create a loss function closure that handles the transformer inputs
                    def loss_fn(outputs, targets):
                        return self.criterion(outputs, targets)
                    
                    # For transformer models, we need to handle the complex input structure
                    # This is a workaround since TALT expects a simpler (x, y) format
                    # We use a wrapper to ensure TALT can handle the transformer model's inputs
                    def model_forward(x):
                        return self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids if token_type_ids is not None else None
                        )
                    
                    # Call TALT's step method with our model forward function
                    loss_val, outputs = self.optimizer.step(
                        lambda x, y: self.criterion(model_forward(x), y),
                        input_ids,  # Placeholder input
                        labels
                    )
                    
                else:  # For CNN models - simple (inputs, labels) format
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Let TALT handle the full forward/backward/optimize process
                    loss_val, outputs = self.optimizer.step(self.criterion, inputs, labels)
                
                # TALT returns the loss value directly, not a tensor
                loss = torch.tensor(loss_val) if not isinstance(loss_val, torch.Tensor) else loss_val
                
            else:  # Standard PyTorch optimizers
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
            
            # Calculate metrics
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