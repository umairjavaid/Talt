"""
Main script for running experiments with the improved TALT optimizer.

Usage:
    python -m examples.main --dataset cifar10 --epochs 10 --use-talt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

import talt
from talt import (
    SimpleCNN,
    Timer,
    print_memory_usage,
    get_loaders,
    train_and_evaluate_improved
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train with Improved TALT Optimizer")
    
    # Dataset and model options
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist'],
                        help='Dataset to use for training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--use-talt', action='store_true',
                        help='Whether to use the improved TALT optimizer')
    
    # TALT-specific parameters
    parser.add_argument('--projection-dim', type=int, default=32,
                        help='Dimension after random projection')
    parser.add_argument('--update-interval', type=int, default=20,
                        help='Steps between topology updates')
    parser.add_argument('--valley-strength', type=float, default=0.2,
                        help='Strength of valley acceleration')
    parser.add_argument('--smoothing-factor', type=float, default=0.3,
                        help='Factor for smoothing high-curvature directions')
    
    # Output parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for datasets')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory for saving results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()

def main():
    """Main function for running training."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Print setup information
    print(f"\n{'='*60}")
    print(f"Starting experiment with {'Improved TALT' if args.use_talt else 'Standard SGD'}")
    print(f"Dataset: {args.dataset}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}, Device: {device}")
    if args.use_talt:
        print(f"TALT parameters:")
        print(f"  - Projection dimension: {args.projection_dim}")
        print(f"  - Update interval: {args.update_interval}")
        print(f"  - Valley strength: {args.valley_strength}")
        print(f"  - Smoothing factor: {args.smoothing_factor}")
    print(f"{'='*60}\n")
    
    # Print CUDA information
    if use_cuda:
        cuda_device = torch.cuda.current_device()
        print(f"Using CUDA device: {torch.cuda.get_device_name(cuda_device)}")
        print(f"CUDA capabilities: {torch.cuda.get_device_capability(cuda_device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(cuda_device).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    optimizer_name = 'talt' if args.use_talt else 'sgd'
    experiment_name = f"{args.dataset}_{optimizer_name}_{timestamp}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parameters for reproducibility
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}={value}\n")
    
    # Load dataset
    print("\nLoading dataset...")
    with Timer("Dataset loading"):
        train_loader, test_loader, num_channels, image_size, num_classes = get_loaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            data_dir=args.data_dir
        )
    
    # Create model
    print("\nCreating model...")
    model = SimpleCNN(
        num_channels=num_channels,
        image_size=image_size,
        num_classes=num_classes
    )
    
    # Print model summary
    print("\nModel architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check memory before training
    print("\nMemory usage before training:")
    print_memory_usage("Initial ")
    
    # Train and evaluate
    print("\nStarting training and evaluation...")
    results = train_and_evaluate_improved(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_improved_talt=args.use_talt,
        device=device,
        projection_dim=args.projection_dim,
        update_interval=args.update_interval,
        valley_strength=args.valley_strength,
        smoothing_factor=args.smoothing_factor,
        visualization_dir=output_dir,
        experiment_name=experiment_name
    )
    
    # Check memory after training
    print("\nMemory usage after training:")
    print_memory_usage("Final ")
    
    # Save results
    print(f"\nExperiment completed. Results saved in {output_dir}")
    print(f"Final test accuracy: {results['final_test_acc']:.2f}%")
    
    return results

if __name__ == '__main__':
    main()
