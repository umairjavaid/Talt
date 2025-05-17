#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(results, output_path):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        results: Dictionary containing training history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(results['train_loss'], label='Training Loss')
    plt.plot(results['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results['train_acc'], label='Training Accuracy')
    plt.plot(results['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_optimizer_comparison(results_dir, experiment_dirs, output_path):
    """
    Plot comparison between different optimizer results.
    
    Args:
        results_dir: Directory containing experiment results
        experiment_dirs: List of experiment directories to compare
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Load results for each experiment
    experiment_results = []
    for exp_dir in experiment_dirs:
        results_path = os.path.join(results_dir, exp_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                experiment_results.append({
                    'name': exp_dir,
                    'optimizer': results['optimizer_type'],
                    'results': results
                })
    
    if not experiment_results:
        print("No experiment results found.")
        return
    
    # Plot validation accuracy comparison
    plt.subplot(2, 2, 1)
    for exp in experiment_results:
        plt.plot(exp['results']['val_acc'], 
                 label=f"{exp['name']} ({exp['optimizer']})")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot validation loss comparison
    plt.subplot(2, 2, 2)
    for exp in experiment_results:
        plt.plot(exp['results']['val_loss'], 
                 label=f"{exp['name']} ({exp['optimizer']})")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot test accuracy comparison as bar chart
    plt.subplot(2, 2, 3)
    names = [exp['name'] for exp in experiment_results]
    test_accs = [exp['results']['test_acc'] for exp in experiment_results]
    colors = sns.color_palette("Set1", len(experiment_results))
    
    bars = plt.bar(names, test_accs, color=colors)
    plt.xlabel('Experiment')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Plot epochs to best validation accuracy
    plt.subplot(2, 2, 4)
    names = [exp['name'] for exp in experiment_results]
    best_epochs = [exp['results']['best_epoch'] for exp in experiment_results]
    
    bars = plt.bar(names, best_epochs, color=colors)
    plt.xlabel('Experiment')
    plt.ylabel('Best Epoch')
    plt.title('Epochs to Best Validation Accuracy')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{int(height)}', ha='center', va='bottom', rotation=0)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_training_report(experiment, output_dir):
    """
    Create visualization report for training results.
    
    Args:
        experiment: Experiment object with results
        output_dir: Directory to save visualizations
    """
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot learning curves
    learning_curves_path = os.path.join(vis_dir, 'learning_curves.png')
    plot_learning_curves(experiment.results, learning_curves_path)
    
    # If TALT-specific visualizations are available (optimizer dependent)
    if experiment.optimizer_type == 'talt' and hasattr(experiment.optimizer, 'get_trajectory_data'):
        try:
            # This is a placeholder for TALT-specific visualizations
            # The actual implementation depends on what the TALT optimizer exposes
            trajectory_data = experiment.optimizer.get_trajectory_data()
            _plot_talt_trajectory(trajectory_data, os.path.join(vis_dir, 'talt_trajectory.png'))
        except:
            print("Could not generate TALT-specific visualizations")
    
    # Architecture-specific visualizations
    try:
        # Get a batch of data
        if experiment.model.model_type == 'cnn':
            # Get the first batch from validation set
            data_iter = iter(experiment.val_loader)
            inputs, _ = next(data_iter)
            inputs = inputs.to(experiment.device)
            
            # Generate architecture-specific visualizations
            vis_data = experiment.model.architecture_specific_visualization(inputs[:4])  # Use first 4 samples
            
            # Plot feature maps
            from .feature_visualization import plot_cnn_feature_maps
            plot_cnn_feature_maps(vis_data, os.path.join(vis_dir, 'feature_maps.png'))
        
        elif experiment.model.model_type == 'llm':
            # Get the first batch from validation set
            data_iter = iter(experiment.val_loader)
            batch = next(data_iter)
            
            # Move to device
            batch = {k: v.to(experiment.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Generate architecture-specific visualizations
            vis_data = experiment.model.architecture_specific_visualization(batch)
            
            # Plot attention maps
            from .attention_visualization import plot_bert_attention
            plot_bert_attention(vis_data, os.path.join(vis_dir, 'attention_maps.png'))
    except Exception as e:
        print(f"Error generating architecture-specific visualizations: {str(e)}")
    
    return vis_dir

def _plot_talt_trajectory(trajectory_data, output_path):
    """
    Plot TALT optimization trajectory data.
    This is a placeholder function for TALT-specific visualizations.
    
    Args:
        trajectory_data: Data from TALT optimizer
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # This is just a placeholder implementation
    # The actual implementation would depend on the TALT optimizer's API
    if 'loss_landscape' in trajectory_data:
        plt.subplot(2, 2, 1)
        plt.imshow(trajectory_data['loss_landscape'], cmap='viridis')
        plt.colorbar()
        plt.title('Loss Landscape')
    
    if 'trajectory_points' in trajectory_data:
        plt.subplot(2, 2, 2)
        points = np.array(trajectory_data['trajectory_points'])
        plt.scatter(points[:, 0], points[:, 1], c=range(len(points)), cmap='viridis')
        plt.colorbar(label='Iteration')
        plt.title('Optimization Trajectory')
    
    if 'valley_strength' in trajectory_data:
        plt.subplot(2, 2, 3)
        plt.plot(trajectory_data['valley_strength'])
        plt.title('Valley Strength')
        plt.xlabel('Iteration')
        plt.ylabel('Strength')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()