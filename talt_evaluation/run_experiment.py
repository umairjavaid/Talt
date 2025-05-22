#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import logging
from datetime import datetime
import sys
import importlib.util

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules using relative or absolute imports as needed
try:
    from talt_evaluation.datasets import get_dataset
    from talt_evaluation.architectures import get_architecture
    from talt_evaluation.hyperparameter_tuning import TaltTuner
    from talt_evaluation.visualization import create_training_report
    from talt_evaluation.experiments import Experiment
except ImportError:
    # If the import fails, try direct imports (for when running from project root)
    from datasets import get_dataset
    from architectures import get_architecture
    from hyperparameter_tuning import TaltTuner
    from visualization import create_training_report
    from experiments import Experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('talt_experiment')

def parse_args():
    parser = argparse.ArgumentParser(description='Run TALT optimization experiments')
    
    # Basic experiment configuration
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--architecture', type=str, required=True, 
                        choices=['resnet18', 'resnet50', 'vgg16', 'efficientnet-b0', 'bert-base'],
                        help='Neural network architecture')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['cifar10', 'cifar100', 'glue-sst2'],
                        help='Dataset to use for training and evaluation')
    parser.add_argument('--optimizer', type=str, required=True, 
                        choices=['talt', 'sgd', 'adam'],
                        help='Optimizer to use for training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    
    # TALT specific parameters
    parser.add_argument('--projection-dim', type=int, default=64, help='TALT projection dimension')
    parser.add_argument('--memory-size', type=int, default=10, help='TALT memory size')
    parser.add_argument('--update-interval', type=int, default=100, help='TALT update interval')
    parser.add_argument('--valley-strength', type=float, default=0.1, help='TALT valley strength')
    parser.add_argument('--smoothing-factor', type=float, default=0.9, help='TALT smoothing factor')
    parser.add_argument('--grad-store-interval', type=int, default=10, help='TALT gradient store interval')
    parser.add_argument('--cov-decay', type=float, default=0.99, help='TALT covariance decay')
    parser.add_argument('--adaptive-reg', type=float, default=1e-5, help='TALT adaptive regularization')
    
    # Hyperparameter tuning
    parser.add_argument('--tune-hyperparams', action='store_true', help='Tune TALT hyperparameters')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of hyperparameter tuning trials')
    parser.add_argument('--study-name', type=str, default=None, help='Optuna study name')
    
    # Experiment output
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--save-checkpoints', action='store_true', help='Save model checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Interval for saving checkpoints')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')
    
    # Hardware configuration
    parser.add_argument('--gpu-index', type=int, default=0, help='GPU index')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(args.output_dir, f"{args.name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Experiment configuration saved to {config_path}")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    train_loader, val_loader, test_loader = get_dataset(
        args.dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get model architecture
    model, model_config = get_architecture(
        args.architecture,
        dataset=args.dataset,
        pretrained=False
    )
    model = model.to(device)
    
    # Configure optimizer parameters
    optimizer_config = {
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    
    # Add TALT specific parameters if applicable
    if args.optimizer == 'talt':
        talt_params = {
            'projection_dim': args.projection_dim,
            'memory_size': args.memory_size,
            'update_interval': args.update_interval,
            'valley_strength': args.valley_strength,
            'smoothing_factor': args.smoothing_factor,
            'grad_store_interval': args.grad_store_interval,
            'cov_decay': args.cov_decay,
            'adaptive_reg': args.adaptive_reg
        }
        optimizer_config.update(talt_params)
    
    if args.tune_hyperparams:
        if args.optimizer != 'talt':
            logger.error("Hyperparameter tuning is only available for TALT optimizer")
            return
        
        # Configure hyperparameter tuning
        study_name = args.study_name or f"{args.architecture}_{args.dataset}_tuning"
        
        tuner = TaltTuner(
            model=model,
            model_config=model_config,
            train_loader=train_loader,
            val_loader=val_loader,
            study_name=study_name,
            output_dir=experiment_dir,
            device=device
        )
        
        # Run hyperparameter tuning
        best_params = tuner.run_study(n_trials=args.n_trials)
        
        # Update optimizer config with best parameters
        optimizer_config.update(best_params)
        logger.info(f"Best hyperparameters: {best_params}")
    
    # Create and run experiment
    experiment = Experiment(
        model=model,
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer_type=args.optimizer,
        optimizer_config=optimizer_config,
        epochs=args.epochs,
        device=device,
        output_dir=experiment_dir,
        mixed_precision=args.mixed_precision,
        save_checkpoints=args.save_checkpoints,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        experiment.load_checkpoint(args.resume_from)
    
    # Run experiment
    experiment.run()
    
    # Create visualization report
    create_training_report(experiment, experiment_dir)
    
    logger.info(f"Experiment completed. Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()