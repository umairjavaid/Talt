"""Training and evaluation utilities for TALT optimizer."""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
import numpy as np

from talt.model import SimpleCNN
from talt.optimizer import ImprovedTALTOptimizer
from talt.visualization import ImprovedTALTVisualizer
from talt.utils import Timer, print_memory_usage, PerformanceTracker

def get_loaders(
    dataset_name: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders for the specified dataset.

    Args:
        dataset_name: Name of dataset ("cifar10" or "mnist")
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        data_dir: Directory to store datasets

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Common transforms
    normalize_transform = []
    
    if dataset_name.lower() == "cifar10":
        # Define transforms for CIFAR-10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load CIFAR-10
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)
        
        num_channels = 3
        image_size = 32
        num_classes = 10
        
    elif dataset_name.lower() == "mnist":
        # Define transforms for MNIST
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
        # Load MNIST
        trainset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform_test)
        
        num_channels = 1
        image_size = 28
        num_classes = 10
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create data loaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size*2, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(trainset)}")
    print(f"Testing samples: {len(testset)}")
    print(f"Num channels: {num_channels}, Image size: {image_size}")
    
    return train_loader, test_loader, num_channels, image_size, num_classes

def train_and_evaluate_improved(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.01,
    use_improved_talt: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    projection_dim: int = 32,
    update_interval: int = 20,
    memory_size: int = 10,
    valley_strength: float = 0.2,
    smoothing_factor: float = 0.3,
    visualization_dir: str = "./visualizations",
    experiment_name: str = "TALT_Experiment"
) -> Dict[str, Any]:
    """
    Train and evaluate a model using the Improved TALT optimizer.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of epochs to train
        learning_rate: Learning rate
        use_improved_talt: Whether to use Improved TALT or standard optimizer
        device: Device to run on
        projection_dim: Dimension after random projection
        update_interval: Steps between topology updates
        memory_size: Number of past gradients to store
        valley_strength: Strength of valley acceleration
        smoothing_factor: Factor for smoothing high-curvature directions
        visualization_dir: Directory for saving visualizations
        experiment_name: Name of the experiment

    Returns:
        Dictionary with training results
    """
    start_time = time.time()
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if use_improved_talt:
        print(f"Using Improved TALT optimizer with projection_dim={projection_dim}, "
              f"update_interval={update_interval}")
        optimizer = ImprovedTALTOptimizer(
            model=model,
            base_optimizer=lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
            lr=learning_rate,
            projection_dim=projection_dim,
            memory_size=memory_size,
            update_interval=update_interval,
            valley_strength=valley_strength,
            smoothing_factor=smoothing_factor,
            device=device
        )
        
        # Ensure scheduler compatibility
        if hasattr(optimizer, 'optimizer') and not hasattr(optimizer, 'param_groups'):
            optimizer.param_groups = optimizer.optimizer.param_groups
    else:
        print(f"Using standard SGD optimizer with lr={learning_rate}")
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer if use_improved_talt else optimizer, 
                                                  T_max=epochs)
    
    # Performance tracking
    tracker = PerformanceTracker()
    visualizer = ImprovedTALTVisualizer(output_dir=visualization_dir)
    
    # Training results
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_times": []
    }
    
    print(f"\nStarting training for {epochs} epochs on {device}")
    print("-" * 60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Record batch start time
            batch_start = time.time()
            
            if use_improved_talt:
                # Forward and backward pass handled by optimizer.step
                with Timer("Batch processing") as timer:
                    loss, outputs = optimizer.step(criterion, inputs, targets)
            else:
                # Standard training loop
                optimizer.zero_grad()
                
                # Forward pass
                forward_start = time.time()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                forward_time = time.time() - forward_start
                tracker.record_timing("forward_pass", forward_time)
                
                # Backward pass
                backward_start = time.time()
                loss.backward()
                backward_time = time.time() - backward_start
                tracker.record_timing("backward_pass", backward_time)
                
                # Optimizer step
                optim_start = time.time()
                optimizer.step()
                optim_time = time.time() - optim_start
                tracker.record_timing("optimizer_step", optim_time)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update statistics
            train_loss += loss
            
            # Record total batch time
            batch_time = time.time() - batch_start
            tracker.record_timing("batch_total", batch_time)
            
            # Record memory usage
            if batch_idx % 20 == 0:
                tracker.record_memory()
            
            # Print progress
            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss:.4f} | Acc: {100.*correct/total:.2f}% | "
                      f"Batch time: {batch_time:.4f}s")
        
        # Compute epoch statistics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update LR scheduler
        scheduler.step()
        
        # Record epoch results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        results["epoch_times"].append(epoch_time)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print("-" * 60)
        
        # Collect visualization data if using improved optimizer
        if use_improved_talt:
            visualizer.add_data({
                'loss_values': optimizer._visualization_data['loss_values'],
                'valley_detections': optimizer._visualization_data['valley_detections'],
                'bifurcations': optimizer.bifurcations,
                'gradient_stats': {
                    name: list(stats) for name, stats in 
                    optimizer._visualization_data['gradient_stats'].items()
                }
            })
    
    # Calculate total training time
    total_time = time.time() - start_time
    
    # Final evaluation
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print("=" * 60)
    
    # Print performance summary
    tracker.print_summary()
    
    # Generate visualizations if using improved optimizer
    if use_improved_talt:
        print("\nGenerating visualizations...")
        visualizer.generate_report(experiment_name=experiment_name)
    
    # Clean up
    if use_improved_talt:
        optimizer.shutdown()
    
    # Return results
    return {
        "train_loss": results["train_loss"],
        "train_acc": results["train_acc"],
        "test_loss": results["test_loss"],
        "test_acc": results["test_acc"],
        "epoch_times": results["epoch_times"],
        "final_test_acc": final_test_acc,
        "total_time": total_time
    }

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: Callable,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[float, float]:
    """
    Evaluate model on test data.

    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on

    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Compute average loss and accuracy
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc
