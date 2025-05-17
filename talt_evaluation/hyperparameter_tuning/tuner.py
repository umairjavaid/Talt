#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

# Setup logging
logger = logging.getLogger(__name__)

class TaltTuner:
    """
    Hyperparameter tuner for TALT optimizer using Optuna.
    """
    
    def __init__(self, model, model_config, train_loader, val_loader, study_name, 
                 output_dir, device, n_epochs=10):
        """
        Initialize the TALT hyperparameter tuner.
        
        Args:
            model: The model to optimize
            model_config: Configuration of the model
            train_loader: Training data loader
            val_loader: Validation data loader
            study_name: Name of the Optuna study
            output_dir: Directory to save results
            device: Device to run on ('cpu' or 'cuda')
            n_epochs: Number of epochs for each trial
        """
        self.model = model
        self.model_config = model_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.study_name = study_name
        self.output_dir = output_dir
        self.device = device
        self.n_epochs = n_epochs
        
        # Create storage path for the study
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup sqlite database for optuna
        self.storage = f"sqlite:///{os.path.join(output_dir, f'{study_name}.db')}"
        
        # Get the parameter search space from the model
        self.search_space = model.get_hyperparameter_search_space()
    
    def _create_optimizer(self, trial):
        """
        Create a TALT optimizer with hyperparameters suggested by Optuna trial.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            TALT optimizer instance
        """
        from talt import TALT  # Assumed TALT optimizer is installed
        
        # Fixed parameters
        lr = 0.1 if self.model.model_type == 'cnn' else 2e-5
        weight_decay = 5e-4 if self.model.model_type == 'cnn' else 0.01
        
        # Sample hyperparameters from search space
        params = {}
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )
            elif param_config['type'] == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
        
        # Create optimizer
        optimizer = TALT(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **params
        )
        
        return optimizer, params
    
    def _train_epoch(self, model, train_loader, optimizer, criterion, device, scaler=None):
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scaler: GradScaler for mixed precision training
        
        Returns:
            float: Average training loss
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different dataset types
            if isinstance(batch, dict):  # For BERT/transformer models
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                _, predicted = torch.max(outputs, 1)
                
            else:  # For CNN models
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                _, predicted = torch.max(outputs, 1)
            
            total_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        return total_loss / len(train_loader), correct / total
    
    def _validate(self, model, val_loader, criterion, device):
        """
        Validate the model.
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function
            device: Device to validate on
        
        Returns:
            tuple: (validation loss, validation accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle different dataset types
                if isinstance(batch, dict):  # For BERT/transformer models
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                else:  # For CNN models
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                total_loss += loss.item()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def objective(self, trial):
        """
        Optuna trial objective function.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            float: Validation accuracy (to be maximized)
        """
        # Reset model weights
        if trial.number > 0:
            for layer in self.model.modules():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        
        # Create optimizer with trial-suggested hyperparameters
        optimizer, params = self._create_optimizer(trial)
        
        # Set loss function based on model type
        if self.model.model_type == 'cnn':
            criterion = torch.nn.CrossEntropyLoss()
        else:  # For BERT
            criterion = torch.nn.CrossEntropyLoss()
        
        # Use mixed precision for faster training
        scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        patience = 3  # Early stopping patience
        
        for epoch in range(self.n_epochs):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(
                self.model, 
                self.train_loader, 
                optimizer, 
                criterion, 
                self.device, 
                scaler
            )
            
            # Validate
            val_loss, val_acc = self._validate(
                self.model, 
                self.val_loader, 
                criterion, 
                self.device
            )
            
            # Report to Optuna
            trial.report(val_acc, epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            logger.info(f"Trial {trial.number}, Epoch {epoch}: "
                        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Check for trial pruning
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned")
                raise optuna.TrialPruned()
        
        return best_val_acc
    
    def run_study(self, n_trials=30):
        """
        Run the hyperparameter optimization study.
        
        Args:
            n_trials: Number of trials to run
        
        Returns:
            dict: Best hyperparameters
        """
        # Create a new study or load existing one
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Save results
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "study_name": self.study_name,
            "model_config": self.model_config,
            "n_trials": n_trials
        }
        
        results_path = os.path.join(self.output_dir, f"{self.study_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {best_value}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Generate visualization of the study
        self._visualize_study(study)
        
        return best_params
    
    def _visualize_study(self, study):
        """
        Create visualizations of the hyperparameter optimization results.
        
        Args:
            study: Optuna study object
        """
        # Create directory for visualizations
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualization 1: Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(viz_dir, "optimization_history.png"))
        
        # Visualization 2: Parameter importances
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(os.path.join(viz_dir, "param_importances.png"))
        except:
            logger.warning("Could not generate parameter importance plot")
        
        # Visualization 3: Slice plot for each parameter
        for param_name in self.search_space.keys():
            try:
                fig = optuna.visualization.plot_slice(study, params=[param_name])
                fig.write_image(os.path.join(viz_dir, f"slice_{param_name}.png"))
            except:
                logger.warning(f"Could not generate slice plot for {param_name}")
        
        # Visualization 4: Parallel coordinate plot
        try:
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(os.path.join(viz_dir, "parallel_coordinate.png"))
        except:
            logger.warning("Could not generate parallel coordinate plot")
        
        logger.info(f"Visualizations saved to {viz_dir}")