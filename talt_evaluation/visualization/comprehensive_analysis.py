#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Analysis Visualizations

Extends existing visualization framework with cross-experiment analysis.
Uses existing plot_learning_curves, plot_optimizer_comparison functions.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple

# Import existing visualization components
from .training_visualization import plot_learning_curves, plot_optimizer_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('talt_comprehensive_analysis')

class CrossExperimentAnalyzer:
    """Analyzes results across multiple experiments using existing viz components."""
    
    def __init__(self, results_directory: Union[str, Path]):
        """
        Initialize cross-experiment analyzer.
        
        Args:
            results_directory: Directory containing experiment results
        """
        self.results_dir = Path(results_directory)
        self.all_results = self._collect_all_results()
        
        # Set visualization styles consistent with existing components
        self.set_visualization_style()
        
    def set_visualization_style(self):
        """Set consistent visualization style."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
    
    def _collect_all_results(self) -> pd.DataFrame:
        """
        Collect all results.json files from experiment directories.
        
        Returns:
            DataFrame containing all experiment results
        """
        all_data = []
        
        # Find all batch directories
        batch_dirs = [d for d in self.results_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for batch_dir in batch_dirs:
            batch_summary_path = batch_dir / "batch_summary.json"
            
            # Process based on batch_summary.json if it exists
            if batch_summary_path.exists():
                try:
                    with open(batch_summary_path, 'r') as f:
                        batch_summary = json.load(f)
                    
                    # Extract batch metadata
                    batch_name = batch_dir.name
                    
                    # Process each experiment in the batch
                    for result in batch_summary.get('results', []):
                        if result.get('success', False):
                            # Find the experiment directory
                            cmd = result.get('cmd', '')
                            # Extract output dir from command if available
                            output_dir = None
                            for part in cmd.split():
                                if part.startswith(str(batch_dir)):
                                    output_dir = Path(part)
                                    break
                            
                            if output_dir is None or not output_dir.exists():
                                # Try to find experiment by success status
                                exp_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
                                for exp_dir in exp_dirs:
                                    results_path = exp_dir / "results.json"
                                    if results_path.exists():
                                        output_dir = exp_dir
                                        break
                            
                            if output_dir and output_dir.exists():
                                # Process the individual experiment results
                                results_path = output_dir / "results.json"
                                if results_path.exists():
                                    with open(results_path, 'r') as f:
                                        exp_results = json.load(f)
                                    
                                    # Extract key metrics
                                    data = {
                                        'batch': batch_name,
                                        'experiment': output_dir.name,
                                        'optimizer': exp_results.get('optimizer_type', 'unknown'),
                                        'model_config': exp_results.get('model_config', {}),
                                        'architecture': exp_results.get('model_config', {}).get('name', 'unknown'),
                                        'dataset': exp_results.get('model_config', {}).get('dataset', 'unknown'),
                                        'test_acc': exp_results.get('test_acc', 0.0),
                                        'test_loss': exp_results.get('test_loss', 0.0),
                                        'best_val_acc': exp_results.get('best_val_acc', 0.0),
                                        'training_time': exp_results.get('training_time', 0.0),
                                        'epochs': len(exp_results.get('train_loss', [])),
                                        'best_epoch': exp_results.get('best_epoch', 0),
                                        'train_loss_history': exp_results.get('train_loss', []),
                                        'val_loss_history': exp_results.get('val_loss', []),
                                        'train_acc_history': exp_results.get('train_acc', []),
                                        'val_acc_history': exp_results.get('val_acc', [])
                                    }
                                    
                                    # Add optimizer-specific parameters
                                    if 'optimizer_config' in exp_results:
                                        for key, value in exp_results['optimizer_config'].items():
                                            data[f'opt_{key}'] = value
                                    
                                    all_data.append(data)
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_dir}: {e}")
            
            else:
                # If no batch summary, look for individual experiment directories
                exp_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
                for exp_dir in exp_dirs:
                    results_path = exp_dir / "results.json"
                    if results_path.exists():
                        try:
                            with open(results_path, 'r') as f:
                                exp_results = json.load(f)
                            
                            # Extract key metrics
                            data = {
                                'batch': batch_dir.name,
                                'experiment': exp_dir.name,
                                'optimizer': exp_results.get('optimizer_type', 'unknown'),
                                'model_config': exp_results.get('model_config', {}),
                                'architecture': exp_results.get('model_config', {}).get('name', 'unknown'),
                                'dataset': exp_results.get('model_config', {}).get('dataset', 'unknown'),
                                'test_acc': exp_results.get('test_acc', 0.0),
                                'test_loss': exp_results.get('test_loss', 0.0),
                                'best_val_acc': exp_results.get('best_val_acc', 0.0),
                                'training_time': exp_results.get('training_time', 0.0),
                                'epochs': len(exp_results.get('train_loss', [])),
                                'best_epoch': exp_results.get('best_epoch', 0),
                                'train_loss_history': exp_results.get('train_loss', []),
                                'val_loss_history': exp_results.get('val_loss', []),
                                'train_acc_history': exp_results.get('train_acc', []),
                                'val_acc_history': exp_results.get('val_acc', [])
                            }
                            
                            # Add optimizer-specific parameters
                            if 'optimizer_config' in exp_results:
                                for key, value in exp_results['optimizer_config'].items():
                                    data[f'opt_{key}'] = value
                            
                            all_data.append(data)
                            
                        except Exception as e:
                            logger.error(f"Error processing experiment {exp_dir}: {e}")
        
        # Create DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Collected results from {len(df)} experiments across {len(batch_dirs)} batches")
            return df
        else:
            logger.warning("No experiment results found")
            return pd.DataFrame()
    
    def generate_optimizer_performance_matrix(self, output_file: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Generate performance matrix across all optimizer/architecture/dataset combinations.
        
        Args:
            output_file: Path to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        if self.all_results.empty:
            logger.warning("No data available to create optimizer performance matrix")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
            
        # Group by architecture, dataset, and optimizer to get mean performance
        perf_data = self.all_results.groupby(['architecture', 'dataset', 'optimizer']).agg({
            'test_acc': 'mean',
            'training_time': 'mean'
        }).reset_index()
        
        # Pivot for heatmap format
        try:
            # Try to create heatmap for test accuracy
            acc_pivot = perf_data.pivot_table(
                index=['architecture', 'dataset'], 
                columns='optimizer', 
                values='test_acc'
            )
            
            # Sort by overall performance
            acc_pivot['mean'] = acc_pivot.mean(axis=1)
            acc_pivot = acc_pivot.sort_values('mean', ascending=False)
            acc_pivot = acc_pivot.drop('mean', axis=1)
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot test accuracy heatmap
            sns.heatmap(
                acc_pivot * 100,  # Convert to percentage
                annot=True, 
                fmt=".2f", 
                cmap="viridis", 
                ax=axes[0],
                cbar_kws={'label': 'Test Accuracy (%)'}
            )
            axes[0].set_title("Test Accuracy (%) by Architecture, Dataset, and Optimizer")
            axes[0].set_ylabel("Architecture / Dataset")
            
            # Try to create heatmap for training time
            time_pivot = perf_data.pivot_table(
                index=['architecture', 'dataset'], 
                columns='optimizer', 
                values='training_time'
            )
            
            # Reindex to match acc_pivot
            time_pivot = time_pivot.reindex(acc_pivot.index)
            
            # Plot training time heatmap
            sns.heatmap(
                time_pivot,
                annot=True, 
                fmt=".1f", 
                cmap="rocket_r",  # Reversed colormap (lower is better)
                ax=axes[1],
                cbar_kws={'label': 'Training Time (s)'}
            )
            axes[1].set_title("Training Time (s) by Architecture, Dataset, and Optimizer")
            axes[1].set_ylabel("Architecture / Dataset")
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating performance matrix: {e}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error creating visualization: {e}", ha='center', va='center')
        
        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved optimizer performance matrix to {output_file}")
        
        return fig
    
    def analyze_convergence_patterns(self, output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Analyze convergence patterns using existing curve plotting.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with convergence analysis results
        """
        if self.all_results.empty:
            logger.warning("No data available to analyze convergence patterns")
            return {}
            
        # Create output directory if needed
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Get unique architectures and datasets
        architectures = self.all_results['architecture'].unique()
        datasets = self.all_results['dataset'].unique()
        optimizers = self.all_results['optimizer'].unique()
        
        results = {
            'architectures': list(architectures),
            'datasets': list(datasets),
            'optimizers': list(optimizers),
            'convergence_metrics': {}
        }
        
        # Analyze convergence for each architecture/dataset combination
        for arch in architectures:
            for dataset in datasets:
                # Filter data for this architecture and dataset
                filtered_data = self.all_results[(self.all_results['architecture'] == arch) & 
                                               (self.all_results['dataset'] == dataset)]
                
                if filtered_data.empty:
                    continue
                
                # Create subplots for this combination
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                fig.suptitle(f"Convergence Analysis: {arch} on {dataset}", fontsize=16)
                
                # Plot validation accuracy curves
                for optimizer in optimizers:
                    opt_data = filtered_data[filtered_data['optimizer'] == optimizer]
                    
                    if opt_data.empty:
                        continue
                    
                    # Use existing frameworks's colors and styles
                    if optimizer == 'talt':
                        color = 'green'
                        linestyle = '-'
                    elif optimizer == 'sgd':
                        color = 'blue'
                        linestyle = '--'
                    elif optimizer == 'adam':
                        color = 'red'
                        linestyle = '-.'
                    else:
                        color = None
                        linestyle = '-'
                    
                    # Plot validation accuracy
                    for _, row in opt_data.iterrows():
                        epochs = list(range(1, len(row['val_acc_history']) + 1))
                        axes[0].plot(epochs, row['val_acc_history'], 
                                    color=color, linestyle=linestyle, alpha=0.3)
                    
                    # Calculate and plot mean validation accuracy
                    avg_val_acc = np.array([opt_data['val_acc_history'].iloc[i][:j] 
                                         for i in range(len(opt_data)) 
                                         for j in range(1, len(opt_data['val_acc_history'].iloc[i]) + 1)])
                    
                    if len(avg_val_acc) > 0:
                        max_len = max(len(row['val_acc_history']) for _, row in opt_data.iterrows())
                        mean_val_acc = []
                        std_val_acc = []
                        
                        for epoch in range(1, max_len + 1):
                            epoch_vals = [row['val_acc_history'][epoch-1] for _, row in opt_data.iterrows() 
                                        if epoch <= len(row['val_acc_history'])]
                            mean_val_acc.append(np.mean(epoch_vals))
                            std_val_acc.append(np.std(epoch_vals))
                        
                        epochs = list(range(1, len(mean_val_acc) + 1))
                        axes[0].plot(epochs, mean_val_acc, color=color, linestyle=linestyle, 
                                    linewidth=2, label=f"{optimizer}")
                        axes[0].fill_between(epochs, 
                                          np.array(mean_val_acc) - np.array(std_val_acc),
                                          np.array(mean_val_acc) + np.array(std_val_acc),
                                          color=color, alpha=0.2)
                        
                        # Store convergence metrics
                        if arch not in results['convergence_metrics']:
                            results['convergence_metrics'][arch] = {}
                        if dataset not in results['convergence_metrics'][arch]:
                            results['convergence_metrics'][arch][dataset] = {}
                        
                        # Calculate metrics
                        results['convergence_metrics'][arch][dataset][optimizer] = {
                            'max_accuracy': max(mean_val_acc),
                            'epochs_to_90pct': next((i+1 for i, acc in enumerate(mean_val_acc) 
                                                 if acc >= 0.9 * max(mean_val_acc)), None),
                            'convergence_rate': np.mean(np.diff(mean_val_acc[:10])) if len(mean_val_acc) >= 10 else None
                        }
                
                # Format validation accuracy plot
                axes[0].set_title(f"Validation Accuracy Convergence")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Validation Accuracy")
                axes[0].grid(True, linestyle='--', alpha=0.7)
                axes[0].legend()
                
                # Plot validation loss curves
                for optimizer in optimizers:
                    opt_data = filtered_data[filtered_data['optimizer'] == optimizer]
                    
                    if opt_data.empty:
                        continue
                    
                    # Use existing frameworks's colors and styles
                    if optimizer == 'talt':
                        color = 'green'
                        linestyle = '-'
                    elif optimizer == 'sgd':
                        color = 'blue'
                        linestyle = '--'
                    elif optimizer == 'adam':
                        color = 'red'
                        linestyle = '-.'
                    else:
                        color = None
                        linestyle = '-'
                    
                    # Plot individual validation losses
                    for _, row in opt_data.iterrows():
                        epochs = list(range(1, len(row['val_loss_history']) + 1))
                        axes[1].plot(epochs, row['val_loss_history'], 
                                    color=color, linestyle=linestyle, alpha=0.3)
                    
                    # Calculate and plot mean validation loss
                    max_len = max(len(row['val_loss_history']) for _, row in opt_data.iterrows())
                    mean_val_loss = []
                    std_val_loss = []
                    
                    for epoch in range(1, max_len + 1):
                        epoch_vals = [row['val_loss_history'][epoch-1] for _, row in opt_data.iterrows() 
                                    if epoch <= len(row['val_loss_history'])]
                        mean_val_loss.append(np.mean(epoch_vals))
                        std_val_loss.append(np.std(epoch_vals))
                    
                    epochs = list(range(1, len(mean_val_loss) + 1))
                    axes[1].plot(epochs, mean_val_loss, color=color, linestyle=linestyle, 
                                linewidth=2, label=f"{optimizer}")
                    axes[1].fill_between(epochs, 
                                      np.array(mean_val_loss) - np.array(std_val_loss),
                                      np.array(mean_val_loss) + np.array(std_val_loss),
                                      color=color, alpha=0.2)
                
                # Format validation loss plot
                axes[1].set_title(f"Validation Loss Convergence")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Validation Loss")
                axes[1].grid(True, linestyle='--', alpha=0.7)
                axes[1].legend()
                
                plt.tight_layout()
                
                # Save figure if output_dir is provided
                if output_dir:
                    filename = f"convergence_{arch}_{dataset}.png"
                    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved convergence plot to {output_dir / filename}")
                
                plt.close(fig)
        
        # Generate summary convergence plot
        self._generate_convergence_summary_plot(results, output_dir)
        
        return results
    
    def _generate_convergence_summary_plot(self, convergence_results: Dict, output_dir: Optional[Path] = None) -> plt.Figure:
        """
        Generate a summary plot of convergence metrics across all experiments.
        
        Args:
            convergence_results: Dictionary with convergence analysis results
            output_dir: Directory to save visualizations
            
        Returns:
            Matplotlib figure
        """
        # Extract convergence metrics into a DataFrame
        rows = []
        for arch in convergence_results['convergence_metrics']:
            for dataset in convergence_results['convergence_metrics'][arch]:
                for optimizer in convergence_results['convergence_metrics'][arch][dataset]:
                    metrics = convergence_results['convergence_metrics'][arch][dataset][optimizer]
                    rows.append({
                        'architecture': arch,
                        'dataset': dataset,
                        'optimizer': optimizer,
                        'max_accuracy': metrics['max_accuracy'],
                        'epochs_to_90pct': metrics['epochs_to_90pct'],
                        'convergence_rate': metrics['convergence_rate']
                    })
        
        if not rows:
            logger.warning("No convergence metrics to plot")
            return None
            
        df = pd.DataFrame(rows)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Convergence Metrics Summary", fontsize=16)
        
        # Plot max accuracy
        sns.barplot(x='optimizer', y='max_accuracy', data=df, ax=axes[0])
        axes[0].set_title("Maximum Validation Accuracy")
        axes[0].set_ylim(0.5, 1.0)
        axes[0].grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Plot epochs to 90% of max accuracy
        sns.barplot(x='optimizer', y='epochs_to_90pct', data=df, ax=axes[1])
        axes[1].set_title("Epochs to 90% of Max Accuracy")
        axes[1].grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Plot convergence rate
        sns.barplot(x='optimizer', y='convergence_rate', data=df, ax=axes[2])
        axes[2].set_title("Initial Convergence Rate")
        axes[2].grid(True, linestyle='--', axis='y', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if output_dir is provided
        if output_dir:
            filename = "convergence_summary.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved convergence summary to {output_dir / filename}")
        
        return fig
    
    def generate_efficiency_analysis(self, output_file: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Analyze training efficiency (accuracy/time tradeoffs).
        
        Args:
            output_file: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        if self.all_results.empty:
            logger.warning("No data available for efficiency analysis")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        try:
            # Create efficiency plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Color map for optimizers
            optimizer_colors = {
                'talt': 'green',
                'sgd': 'blue',
                'adam': 'red',
            }
            
            # Shape map for architectures
            architecture_markers = {
                'resnet18': 'o',
                'resnet50': 's',
                'vgg16': '^',
                'efficientnet-b0': 'P',
                'bert-base': 'X',
            }
            
            # Plot each experiment as a point
            for optimizer in self.all_results['optimizer'].unique():
                opt_data = self.all_results[self.all_results['optimizer'] == optimizer]
                
                for arch in opt_data['architecture'].unique():
                    arch_data = opt_data[opt_data['architecture'] == arch]
                    
                    # Plot points
                    color = optimizer_colors.get(optimizer, 'gray')
                    marker = architecture_markers.get(arch, 'o')
                    
                    ax.scatter(
                        arch_data['training_time'],
                        arch_data['test_acc'],
                        color=color,
                        marker=marker,
                        s=100,
                        alpha=0.7,
                        label=f"{optimizer} - {arch}"
                    )
            
            # Add efficiency frontier
            frontier_data = self.all_results.copy()
            frontier_data = frontier_data.sort_values('training_time')
            frontier = []
            best_acc = 0
            
            for _, row in frontier_data.iterrows():
                if row['test_acc'] > best_acc:
                    frontier.append((row['training_time'], row['test_acc']))
                    best_acc = row['test_acc']
            
            if frontier:
                frontier_x, frontier_y = zip(*frontier)
                ax.plot(frontier_x, frontier_y, 'k--', alpha=0.5, label='Efficiency Frontier')
            
            # Format plot
            ax.set_title("Accuracy vs. Training Time Trade-off")
            ax.set_xlabel("Training Time (s)")
            ax.set_ylabel("Test Accuracy")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add annotations for best points
            best_acc_idx = self.all_results['test_acc'].idxmax()
            fastest_high_acc_idx = self.all_results[self.all_results['test_acc'] > 0.8 * self.all_results['test_acc'].max()]['training_time'].idxmin()
            
            if not pd.isna(best_acc_idx):
                best_row = self.all_results.loc[best_acc_idx]
                ax.annotate(
                    f"Best Accuracy: {best_row['optimizer']} - {best_row['architecture']}",
                    xy=(best_row['training_time'], best_row['test_acc']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
                )
            
            if not pd.isna(fastest_high_acc_idx):
                fastest_row = self.all_results.loc[fastest_high_acc_idx]
                ax.annotate(
                    f"Fastest High-Accuracy: {fastest_row['optimizer']} - {fastest_row['architecture']}",
                    xy=(fastest_row['training_time'], fastest_row['test_acc']),
                    xytext=(10, -30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
                )
            
            # Handle legend with many entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='lower right')
            
        except Exception as e:
            logger.error(f"Error creating efficiency analysis: {e}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error creating visualization: {e}", ha='center', va='center')
        
        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved efficiency analysis to {output_file}")
        
        return fig

    def generate_ablation_analysis(self, output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Generate ablation study analysis.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with ablation analysis results
        """
        if self.all_results.empty:
            logger.warning("No data available for ablation analysis")
            return {}
            
        # Create output directory if needed
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Find experiments that are part of ablation studies
        # Look for experiment names containing "no_valley", "no_smoothing", etc.
        ablation_keywords = ["no_valley", "no_smoothing", "small_proj", "large_proj", "no_adaptive_reg"]
        ablation_exps = self.all_results[
            self.all_results['experiment'].apply(lambda x: any(kw in x for kw in ablation_keywords))
        ]
        
        if ablation_exps.empty:
            logger.warning("No ablation studies found in the experiment results")
            return {}
        
        results = {
            'ablation_analysis': {}
        }
        
        # Group by architecture and dataset
        for arch in ablation_exps['architecture'].unique():
            for dataset in ablation_exps[ablation_exps['architecture'] == arch]['dataset'].unique():
                arch_data = ablation_exps[(ablation_exps['architecture'] == arch) & 
                                       (ablation_exps['dataset'] == dataset)]
                
                # Extract baseline TALT experiment
                baseline = self.all_results[
                    (self.all_results['architecture'] == arch) & 
                    (self.all_results['dataset'] == dataset) & 
                    (self.all_results['optimizer'] == 'talt') & 
                    ~self.all_results['experiment'].apply(lambda x: any(kw in x for kw in ablation_keywords))
                ]
                
                if baseline.empty:
                    logger.warning(f"No baseline TALT experiment found for {arch} on {dataset}")
                    continue
                
                baseline = baseline.iloc[0]
                
                # Collect ablation data
                ablation_data = []
                for _, row in arch_data.iterrows():
                    # Determine ablation type
                    ablation_type = next((kw for kw in ablation_keywords if kw in row['experiment']), "unknown")
                    
                    # Calculate impact
                    acc_impact = (row['test_acc'] - baseline['test_acc']) / baseline['test_acc'] * 100
                    time_impact = (row['training_time'] - baseline['training_time']) / baseline['training_time'] * 100
                    
                    ablation_data.append({
                        'ablation': ablation_type,
                        'test_acc': row['test_acc'],
                        'training_time': row['training_time'],
                        'acc_impact': acc_impact,
                        'time_impact': time_impact
                    })
                
                # Store in results
                if arch not in results['ablation_analysis']:
                    results['ablation_analysis'][arch] = {}
                
                results['ablation_analysis'][arch][dataset] = {
                    'baseline': {
                        'test_acc': baseline['test_acc'],
                        'training_time': baseline['training_time']
                    },
                    'ablations': ablation_data
                }
                
                # Create visualization
                if ablation_data:
                    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                    fig.suptitle(f"Ablation Study: {arch} on {dataset}", fontsize=16)
                    
                    # Prepare data
                    ablation_df = pd.DataFrame(ablation_data)
                    
                    # Plot accuracy impact
                    sns.barplot(x='ablation', y='acc_impact', data=ablation_df, ax=axes[0])
                    axes[0].set_title("Impact on Test Accuracy (%)")
                    axes[0].set_ylabel("Relative Change (%)")
                    axes[0].axhline(y=0, color='r', linestyle='-')
                    axes[0].grid(True, linestyle='--', axis='y', alpha=0.7)
                    
                    # Plot time impact
                    sns.barplot(x='ablation', y='time_impact', data=ablation_df, ax=axes[1])
                    axes[1].set_title("Impact on Training Time (%)")
                    axes[1].set_ylabel("Relative Change (%)")
                    axes[1].axhline(y=0, color='r', linestyle='-')
                    axes[1].grid(True, linestyle='--', axis='y', alpha=0.7)
                    
                    plt.tight_layout()
                    
                    # Save figure if output_dir is provided
                    if output_dir:
                        filename = f"ablation_{arch}_{dataset}.png"
                        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
                        logger.info(f"Saved ablation analysis to {output_dir / filename}")
                    
                    plt.close(fig)
        
        return results
