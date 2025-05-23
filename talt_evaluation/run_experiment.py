#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import logging
from datetime import datetime
import sys
import random
import numpy as np

# Get the absolute path to the project root
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Ensure the project root is in the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add the talt_evaluation directory to path
talt_eval_dir = os.path.dirname(script_path)
if talt_eval_dir not in sys.path:
    sys.path.insert(0, talt_eval_dir)

# Import project modules with better error handling
try:
    # Try relative imports first
    try:
        from .datasets import get_dataset
        from .models import get_architecture
        from .hyperparameter_tuning import TaltTuner
        from .visualization import create_training_report
        from .experiments import Experiment
    except ImportError:
        # Fallback to direct imports
        from datasets import get_dataset
        from models import get_architecture
        from hyperparameter_tuning import TaltTuner
        from visualization import create_training_report
        from experiments import Experiment
except ImportError as e:
    # Try importing from talt_evaluation package
    try:
        from talt_evaluation.datasets import get_dataset
        from talt_evaluation.models import get_architecture
        from talt_evaluation.hyperparameter_tuning import TaltTuner
        from talt_evaluation.visualization import create_training_report
        from talt_evaluation.experiments import Experiment
    except ImportError as e2:
        missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
        print(f"Error: Missing required module '{missing_module}'. Please check your installation.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Project root: {project_root}")
        print(f"Python path: {sys.path}")
        print(f"Primary error: {e}")
        print(f"Secondary error: {e2}")
        
        # List available files for debugging
        print("\nAvailable files in talt_evaluation directory:")
        for file in os.listdir(talt_eval_dir):
            print(f"  {file}")
        
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('talt_experiment')

def set_reproducibility(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_experiment_config(args):
    """Validate experiment configuration before running."""
    required_fields = ['name', 'architecture', 'dataset', 'optimizer', 'epochs']
    for field in required_fields:
        if not hasattr(args, field) or getattr(args, field) is None:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate optimizer-specific parameters
    if args.optimizer == 'talt':
        talt_params = ['projection_dim', 'valley_strength', 'smoothing_factor']
        for param in talt_params:
            if not hasattr(args, param) or getattr(args, param) is None:
                logger.warning(f"Missing TALT parameter {param}, using default")
    
    # Additional validation
    if args.lr <= 0:
        raise ValueError("Learning rate must be greater than 0.")

def parse_args():
    parser = argparse.ArgumentParser(description='Run single TALT optimization experiment')
    
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--architecture', type=str, required=True, 
                        choices=['resnet18', 'resnet50', 'vgg16', 'efficientnet-b0', 'bert-base', 'simplecnn'],
                        help='Neural network architecture')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['cifar10', 'cifar100', 'glue-sst2', 'mnist'],
                        help='Dataset to use for training and evaluation')
    parser.add_argument('--optimizer', type=str, required=True, 
                        choices=['talt', 'sgd', 'adam'],
                        help='Optimizer to use for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--projection-dim', type=int, default=64, help='TALT projection dimension')
    parser.add_argument('--memory-size', type=int, default=10, help='TALT memory size')
    parser.add_argument('--update-interval', type=int, default=100, help='TALT update interval')
    parser.add_argument('--valley-strength', type=float, default=0.1, help='TALT valley strength')
    parser.add_argument('--smoothing-factor', type=float, default=0.9, help='TALT smoothing factor')
    parser.add_argument('--grad-store-interval', type=int, default=10, help='TALT gradient store interval')
    parser.add_argument('--cov-decay', type=float, default=0.99, help='TALT covariance decay')
    parser.add_argument('--adaptive-reg', type=float, default=1e-5, help='TALT adaptive regularization')
    parser.add_argument('--tune-hyperparams', action='store_true', help='Tune TALT hyperparameters')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of hyperparameter tuning trials')
    parser.add_argument('--study-name', type=str, default=None, help='Optuna study name')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--save-checkpoints', action='store_true', help='Save model checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Interval for saving checkpoints')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--gpu-index', type=int, default=0, help='GPU index')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def find_latest_checkpoint(experiment_dir):
    """Find the latest checkpoint if experiment was interrupted."""
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None
    
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    latest = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    return latest

def main():
    args = parse_args()
    
    # Set reproducibility
    set_reproducibility(args.seed)
    
    # Validate configuration
    try:
        validate_experiment_config(args)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
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
        'weight_decay': args.weight_decay,
        'momentum': 0.9  # Adding default momentum
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
            'adaptive_reg': args.adaptive_reg,
            'device': device  # Keep device for TALT creation, will be filtered out during JSON serialization
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
        optimizer_config={k: v for k, v in optimizer_config.items() if k != 'device'},  # Remove device from config
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
    
    # Auto-recovery from checkpoints if experiment directory exists
    if args.resume_from is None and os.path.exists(experiment_dir):
        latest_checkpoint = find_latest_checkpoint(experiment_dir)
        if latest_checkpoint:
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            args.resume_from = latest_checkpoint
    
    # Run experiment
    experiment.run()
    
    # Create visualization report
    create_training_report(experiment, experiment_dir)
    
    logger.info(f"Experiment completed. Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()