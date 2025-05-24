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
    Create visualization report for training results with TensorBoard integration.
    
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
    if experiment.optimizer_type in ['talt', 'improved_talt', 'improved-talt', 'original_talt', 'original-talt'] and hasattr(experiment, 'optimizer'):
        try:
            # Run diagnostics if available
            if hasattr(experiment.optimizer, 'diagnose_visualization_state'):
                print("Running TALT diagnostics for visualization report...")
                experiment.optimizer.diagnose_visualization_state()
            
            # Force topology update if needed to ensure data
            if hasattr(experiment.optimizer, 'force_topology_update'):
                experiment.optimizer.force_topology_update()
            
            # Get trajectory data - try TensorBoard-compatible method first
            trajectory_data = None
            if hasattr(experiment.optimizer, 'get_tensorboard_metrics'):
                trajectory_data = experiment.optimizer.get_tensorboard_metrics()
                # Also get historical data
                if hasattr(experiment.optimizer, 'get_visualization_data'):
                    historical_data = experiment.optimizer.get_visualization_data()
                    # Merge with current metrics
                    trajectory_data.update(historical_data)
            elif hasattr(experiment.optimizer, 'get_visualization_data'):
                trajectory_data = experiment.optimizer.get_visualization_data()
            
            if trajectory_data:
                _plot_talt_trajectory(trajectory_data, os.path.join(vis_dir, 'talt_trajectory.png'))
            
        except Exception as e:
            print(f"Could not generate TALT-specific visualizations: {e}")
    
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
    Plot TALT optimization trajectory data with enhanced visualizations.
    
    Args:
        trajectory_data: Data from TALT optimizer
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(15, 10))
    
    # Loss trajectory with valley detections
    if 'loss_history' in trajectory_data or 'loss_values' in trajectory_data:
        plt.subplot(2, 3, 1)
        loss_data = trajectory_data.get('loss_history', trajectory_data.get('loss_values', []))
        if loss_data:
            plt.plot(loss_data, 'b-', linewidth=1.5, label='Loss')
            
            # Mark valley detections
            valley_detections = trajectory_data.get('valley_detections', [])
            bifurcations = trajectory_data.get('bifurcations', [])
            
            for i, detection in enumerate(valley_detections):
                if isinstance(detection, (list, tuple)) and len(detection) >= 2:
                    step = detection[0]
                    if step < len(loss_data):
                        label = 'Valley Detection' if i == 0 else None
                        plt.axvline(x=step, color='red', linestyle='--', alpha=0.7, label=label)
            
            for i, step in enumerate(bifurcations):
                if step < len(loss_data):
                    label = 'Bifurcation' if i == 0 else None
                    plt.axvline(x=step, color='green', linestyle=':', alpha=0.7, label=label)
            
            plt.title('Loss Trajectory with TALT Events')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # Eigenvalue evolution
    if 'eigenvalues' in trajectory_data or 'eigenvalues_history' in trajectory_data:
        plt.subplot(2, 3, 2)
        eigenvalue_data = trajectory_data.get('eigenvalues', trajectory_data.get('eigenvalues_history', {}))
        
        if eigenvalue_data:
            param_count = 0
            for param_name, eig_data in eigenvalue_data.items():
                if param_count >= 3:  # Limit to 3 parameters for clarity
                    break
                    
                if isinstance(eig_data, list) and len(eig_data) > 0:
                    # Handle different formats
                    if isinstance(eig_data[0], tuple):
                        # Format: [(step, eigenvalues), ...]
                        steps = [item[0] for item in eig_data]
                        eigenvals = [item[1] for item in eig_data]
                    elif isinstance(eig_data, dict) and 'steps' in eig_data:
                        # Format: {'steps': [...], 'eigenvalues': [...]}
                        steps = eig_data['steps']
                        eigenvals = eig_data['eigenvalues']
                    else:
                        continue
                    
                    # Plot top eigenvalue
                    if eigenvals and len(eigenvals) > 0:
                        top_eigenvals = []
                        for ev_list in eigenvals:
                            if isinstance(ev_list, (list, np.ndarray)) and len(ev_list) > 0:
                                top_eigenvals.append(ev_list[0])
                            elif isinstance(ev_list, (int, float)):
                                top_eigenvals.append(ev_list)
                        
                        if top_eigenvals:
                            plt.plot(steps[:len(top_eigenvals)], top_eigenvals, 
                                   label=f'{param_name[:10]}...', linewidth=1.5)
                            param_count += 1
        
        plt.title('Top Eigenvalue Evolution')
        plt.xlabel('Step')
        plt.ylabel('Eigenvalue')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Gradient norm evolution
    if 'grad_memory' in trajectory_data or 'gradient_norms_history' in trajectory_data:
        plt.subplot(2, 3, 3)
        
        grad_data = trajectory_data.get('gradient_norms_history', trajectory_data.get('grad_memory', {}))
        
        if grad_data:
            param_count = 0
            for param_name, data in grad_data.items():
                if param_count >= 3:  # Limit to 3 parameters
                    break
                
                if isinstance(data, dict) and 'grad_norms' in data:
                    # Format: {'grad_norms': [...], 'steps': [...]}
                    grad_norms = data['grad_norms']
                    steps = data.get('steps', list(range(len(grad_norms))))
                elif isinstance(data, list) and len(data) > 0:
                    # Format: [(step, grad_norm, ...), ...]
                    if isinstance(data[0], tuple):
                        steps = [item[0] for item in data]
                        grad_norms = [item[1] for item in data]
                    else:
                        grad_norms = data
                        steps = list(range(len(grad_norms)))
                else:
                    continue
                
                if grad_norms:
                    plt.plot(steps[:len(grad_norms)], grad_norms, 
                           label=f'{param_name[:10]}...', linewidth=1.5)
                    param_count += 1
        
        plt.title('Gradient Norm Evolution')
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Valley detection frequency
    if 'valley_detections' in trajectory_data:
        plt.subplot(2, 3, 4)
        detections = trajectory_data['valley_detections']
        
        if detections:
            # Count detections per parameter
            param_counts = {}
            for detection in detections:
                if isinstance(detection, (list, tuple)) and len(detection) >= 2:
                    param_name = detection[1] if len(detection) > 1 else 'unknown'
                    param_counts[param_name] = param_counts.get(param_name, 0) + 1
            
            if param_counts:
                params = list(param_counts.keys())
                counts = list(param_counts.values())
                
                plt.bar(range(len(params)), counts)
                plt.title('Valley Detections by Parameter')
                plt.xlabel('Parameter')
                plt.ylabel('Detection Count')
                plt.xticks(range(len(params)), [p[:10] + '...' if len(p) > 10 else p for p in params], 
                          rotation=45, ha='right')
    
    # Gradient statistics summary
    if 'gradient_stats' in trajectory_data:
        plt.subplot(2, 3, 5)
        grad_stats = trajectory_data['gradient_stats']
        
        if grad_stats:
            param_names = []
            avg_grad_norms = []
            
            for param_name, stats_list in grad_stats.items():
                if isinstance(stats_list, list) and len(stats_list) > 0:
                    # Extract average gradient norm
                    grad_norms = []
                    for stat_entry in stats_list:
                        if isinstance(stat_entry, dict) and 'grad_norm' in stat_entry:
                            grad_norms.append(stat_entry['grad_norm'])
                    
                    if grad_norms:
                        param_names.append(param_name[:10] + '...' if len(param_name) > 10 else param_name)
                        avg_grad_norms.append(np.mean(grad_norms))
            
            if param_names and avg_grad_norms:
                plt.bar(range(len(param_names)), avg_grad_norms)
                plt.title('Average Gradient Norms')
                plt.xlabel('Parameter')
                plt.ylabel('Average Gradient Norm')
                plt.xticks(range(len(param_names)), param_names, rotation=45, ha='right')
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary text
    summary_text = "TALT Summary Statistics:\n\n"
    
    loss_data = trajectory_data.get('loss_history', trajectory_data.get('loss_values', []))
    if loss_data:
        summary_text += f"Total Steps: {len(loss_data)}\n"
        summary_text += f"Final Loss: {loss_data[-1]:.6f}\n"
        summary_text += f"Min Loss: {min(loss_data):.6f}\n"
    
    valley_detections = trajectory_data.get('valley_detections', [])
    bifurcations = trajectory_data.get('bifurcations', [])
    
    summary_text += f"Valley Detections: {len(valley_detections)}\n"
    summary_text += f"Bifurcations: {len(bifurcations)}\n"
    
    eigenvalue_data = trajectory_data.get('eigenvalues', trajectory_data.get('eigenvalues_history', {}))
    summary_text += f"Parameters with Eigendata: {len(eigenvalue_data)}\n"
    
    grad_data = trajectory_data.get('gradient_norms_history', trajectory_data.get('grad_memory', {}))
    summary_text += f"Parameters with Grad History: {len(grad_data)}\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"TALT trajectory visualization saved to {output_path}")