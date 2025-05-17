"""Visualization module for the TALT optimizer."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Any, Optional
import os
import matplotlib.animation as animation
from collections import deque

class ImprovedTALTVisualizer:
    """
    Visualization tools for improved TALT optimizer.

    Provides methods to visualize:
    1. Loss trajectories
    2. Gradient landscapes
    3. Eigenvalue spectra
    4. Valley detection events
    5. Parameter space projections
    """

    def __init__(self, output_dir: str = "./visualizations", max_points: int = 1000):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving visualizations
            max_points: Maximum number of data points to display
        """
        self.output_dir = output_dir
        self.max_points = max_points
        self.data = {
            'loss_values': deque(maxlen=max_points),
            'valley_detections': [],
            'gradient_stats': {},
            'bifurcations': [],
            'parameter_snapshots': []
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Configure plotting style
        sns.set_style("darkgrid")
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14
        })

    def add_data(self, optimizer_data: Dict[str, Any]) -> None:
        """
        Add optimizer data to visualization buffer.

        Args:
            optimizer_data: Dictionary containing optimizer data
        """
        if 'loss_values' in optimizer_data:
            for val in optimizer_data['loss_values']:
                self.data['loss_values'].append(val)

        if 'valley_detections' in optimizer_data:
            self.data['valley_detections'].extend(optimizer_data['valley_detections'])

        if 'bifurcations' in optimizer_data:
            self.data['bifurcations'] = optimizer_data['bifurcations']

        if 'gradient_stats' in optimizer_data:
            for name, stats in optimizer_data['gradient_stats'].items():
                if name not in self.data['gradient_stats']:
                    self.data['gradient_stats'][name] = []
                self.data['gradient_stats'][name].extend(stats)

    def visualize_loss_trajectory(self, 
                                save_path: Optional[str] = None, 
                                show: bool = True) -> None:
        """
        Visualize loss trajectory with valley detection events.

        Args:
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        if not self.data['loss_values']:
            print("No loss data available for visualization")
            return

        plt.figure(figsize=(14, 8))
        
        # Plot loss curve
        loss_values = list(self.data['loss_values'])
        epochs = np.arange(len(loss_values))
        plt.plot(epochs, loss_values, 'b-', linewidth=2, alpha=0.7, label='Training Loss')

        # Add rolling average
        if len(loss_values) > 10:
            window_size = min(20, len(loss_values) // 5)
            rolling_mean = np.convolve(
                loss_values, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            offset = (len(loss_values) - len(rolling_mean)) // 2
            plt.plot(
                epochs[offset:offset+len(rolling_mean)],
                rolling_mean,
                'r-',
                linewidth=3,
                label=f'Rolling Average (window={window_size})'
            )

        # Mark bifurcation events
        for step in self.data['bifurcations']:
            if step < len(loss_values):
                plt.axvline(x=step, color='g', linestyle='--', alpha=0.5)
        
        # Mark valley detection events with stars
        valley_steps = [step for step, _, _ in self.data['valley_detections']]
        if valley_steps:
            valley_steps = [step for step in valley_steps if step < len(loss_values)]
            valley_losses = [loss_values[step] for step in valley_steps]
            plt.scatter(valley_steps, valley_losses, marker='*', s=200, 
                        color='orange', label='Valley Detection', zorder=5)

        # Add labels and legend
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Loss Trajectory with Valley Detections')
        plt.legend(loc='upper right')
        
        # Add grid and limit y-axis if needed
        plt.grid(True, alpha=0.3)
        
        # Dynamically set y limit to exclude outliers but include most data
        if len(loss_values) > 10:
            sorted_loss = sorted(loss_values)
            plt.ylim(sorted_loss[0], sorted_loss[int(0.95*len(sorted_loss))])
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Loss trajectory saved to {full_path}")
            
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_eigenvalue_spectra(self, 
                                   parameter_name: str, 
                                   save_path: Optional[str] = None, 
                                   show: bool = True) -> None:
        """
        Visualize eigenvalue spectra evolution for a parameter.

        Args:
            parameter_name: Name of parameter to visualize
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        if parameter_name not in self.data['gradient_stats']:
            print(f"No eigenvalue data available for parameter {parameter_name}")
            return
            
        param_data = self.data['gradient_stats'][parameter_name]
        if not param_data:
            print(f"Empty eigenvalue data for parameter {parameter_name}")
            return
            
        # Extract data
        steps = [d['step'] for d in param_data]
        eigenvalue_data = [d['eigenvalues'] for d in param_data]
        
        # Determine number of eigenvalues
        n_eigenvalues = min(3, len(eigenvalue_data[0]))
        
        plt.figure(figsize=(14, 8))
        
        # Plot each eigenvalue trajectory
        for i in range(n_eigenvalues):
            values = [ev[i] if i < len(ev) else np.nan for ev in eigenvalue_data]
            plt.plot(steps, values, linewidth=2, 
                    label=f'Eigenvalue {i+1}', marker='o', markersize=4)
        
        # Add labels and legend
        plt.xlabel('Training Steps')
        plt.ylabel('Eigenvalue Magnitude')
        plt.title(f'Eigenvalue Spectrum Evolution for {parameter_name}')
        plt.legend(loc='upper right')
        
        # Add log scale if values span multiple orders of magnitude
        values_flat = [v for ev in eigenvalue_data for v in ev]
        if max(values_flat) / (min(values_flat) + 1e-10) > 100:
            plt.yscale('log')
            
        plt.grid(True, alpha=0.3)
            
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Eigenvalue visualization saved to {full_path}")
            
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_gradient_norm_history(self, 
                                      save_path: Optional[str] = None, 
                                      show: bool = True) -> None:
        """
        Visualize gradient norm history for all parameters.

        Args:
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        if not self.data['gradient_stats']:
            print("No gradient statistics available for visualization")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Plot gradient norms for each parameter
        for name, stats in self.data['gradient_stats'].items():
            if not stats:
                continue
                
            steps = [d['step'] for d in stats]
            norms = [d['grad_norm'] for d in stats]
            
            # Get shorter parameter name for legend
            short_name = name.split('.')[-1]
            plt.plot(steps, norms, linewidth=2, alpha=0.7, 
                    label=f'{short_name}', marker='.', markersize=3)
        
        # Add labels and legend
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm History')
        
        # Use log scale for better visibility
        plt.yscale('log')
        
        # Create compact legend with smaller font
        plt.legend(loc='upper right', fontsize='small', 
                 ncol=min(3, len(self.data['gradient_stats'])))
        
        plt.grid(True, alpha=0.3)
            
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Gradient norm visualization saved to {full_path}")
            
        if show:
            plt.show()
        else:
            plt.close()

    def create_animation(self, fps: int = 5, save_path: str = "loss_animation.mp4") -> None:
        """
        Create an animation of the loss landscape.

        Args:
            fps: Frames per second
            save_path: Path to save the animation
        """
        if len(self.data['loss_values']) < 10:
            print("Not enough data points for animation")
            return

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Trajectory Animation')
        ax.grid(True, alpha=0.3)

        # Prepare data
        loss_values = list(self.data['loss_values'])
        total_steps = len(loss_values)
        
        # Set reasonable y limits
        sorted_losses = sorted(loss_values)
        y_min = sorted_losses[0]
        y_max = sorted_losses[int(0.95*len(sorted_losses))]
        ax.set_ylim(y_min, y_max)

        # Animation function
        def animate(i):
            show_steps = min(total_steps, 50)  # Show at most 50 steps at a time
            start = max(0, i - show_steps)
            end = i + 1

            # Clear and redraw
            ax.clear()
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Trajectory Animation')
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)

            # Plot loss curve
            steps = list(range(start, end))
            values = loss_values[start:end]
            ax.plot(steps, values, 'b-', linewidth=2)

            # Mark current point
            ax.plot([i], [loss_values[i]], 'ro', markersize=8)

            # Show valley detections
            valleys = [(step, self.data['loss_values'][step]) 
                      for step, _, _ in self.data['valley_detections'] 
                      if start <= step < end]
            
            if valleys:
                valley_steps, valley_losses = zip(*valleys)
                ax.scatter(valley_steps, valley_losses, marker='*', s=200, 
                          color='orange', zorder=5)

            # Set x-axis limits to show the right window
            ax.set_xlim(start, max(end, start + show_steps))

        # Create animation
        frames = min(500, total_steps)  # Limit to 500 frames for performance
        step_size = max(1, total_steps // frames)
        frames_indices = list(range(0, total_steps, step_size))
        
        anim = animation.FuncAnimation(
            fig, animate, frames=frames_indices,
            interval=1000/fps, blit=False
        )

        # Save animation
        full_path = os.path.join(self.output_dir, save_path)
        anim.save(full_path, writer='ffmpeg', fps=fps)
        plt.close()
        print(f"Animation saved to {full_path}")

    def generate_report(self, 
                   experiment_name: str = "TALT_Experiment",
                   include_animations: bool = False,
                   include_advanced_visualizations: bool = True) -> None:
        """
        Generate a comprehensive visualization report including all available visualizations.
        
        Args:
            experiment_name: Name of the experiment
            include_animations: Whether to include animations
            include_advanced_visualizations: Whether to include advanced visualizations
        """
        # Create experiment directory
        exp_dir = os.path.join(self.output_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"Generating visualization report in {exp_dir}...")
        
        # Generate basic visualizations
        self.visualize_loss_trajectory(
            save_path=os.path.join(exp_dir, "loss_trajectory.png"),
            show=False
        )
        
        for param_name in self.data['gradient_stats'].keys():
            short_name = param_name.split('.')[-1]
            self.visualize_eigenvalue_spectra(
                parameter_name=param_name,
                save_path=os.path.join(exp_dir, f"eigenvalues_{short_name}.png"),
                show=False
            )
        
        self.visualize_gradient_norm_history(
            save_path=os.path.join(exp_dir, "gradient_norms.png"),
            show=False
        )
        
        # Generate advanced visualizations if requested
        if include_advanced_visualizations:
            try:
                self.visualize_lr_schedule(
                    save_path=os.path.join(exp_dir, "lr_schedule.png"),
                    show=False
                )
                
                # Generate gradient distribution plots for a subset of parameters
                for param_name in list(self.data['gradient_stats'].keys())[:2]:  # Limit to 2 parameters
                    short_name = param_name.split('.')[-1]
                    self.visualize_gradient_distribution(
                        parameter_name=param_name,
                        save_path=os.path.join(exp_dir, f"gradient_dist_{short_name}.png"),
                        show=False
                    )
                
                self.visualize_convergence_vs_time(
                    save_path=os.path.join(exp_dir, "convergence_vs_time.png"),
                    show=False
                )
                
                # Note: multi-run comparisons require multiple run data to be available
                if hasattr(self, 'multi_run_data') and self.multi_run_data:
                    self.visualize_multi_run_comparison(
                        run_results=self.multi_run_data,
                        save_path=os.path.join(exp_dir, "multi_run_comparison.png"),
                        show=False
                    )
                    
                    self.visualize_time_to_threshold(
                        run_results=self.multi_run_data,
                        save_path=os.path.join(exp_dir, "time_to_threshold.png"),
                        show=False
                    )
            except Exception as e:
                print(f"Warning: Some advanced visualizations could not be generated: {e}")
        
        # Generate animations if requested
        if include_animations:
            self.create_animation(
                save_path=os.path.join(exp_dir, "loss_animation.mp4")
            )
        
        print(f"Visualization report generated in {exp_dir}")

    def visualize_lr_schedule(self, 
                             save_path: Optional[str] = None, 
                             show: bool = True) -> None:
        """
        Visualize learning rate schedule overlaid on the loss curve.
        
        This visualization helps correlate learning rate changes with performance jumps.
        
        Args:
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Primary axis for loss
        ax1 = plt.gca()
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.plot(self.data['steps'], self.data['train_loss'], 'b-', label='Training Loss')
        
        # Secondary axis for learning rate
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning Rate', color='r')
        ax2.plot(self.data['steps'], self.data['learning_rates'], 'r-', label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('Learning Rate Schedule and Loss Trajectory')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_gradient_distribution(self,
                                       parameter_name: str,
                                       steps: List[int] = None,
                                       save_path: Optional[str] = None,
                                       show: bool = True) -> None:
        """
        Visualize the distribution of gradient components at selected steps.
        
        This helps detect skewness or heavy tails in the gradient distribution.
        
        Args:
            parameter_name: Name of parameter to visualize
            steps: List of steps at which to plot distributions (default: evenly spaced steps)
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        if parameter_name not in self.data['gradient_stats']:
            raise ValueError(f"Parameter {parameter_name} not found in gradient statistics")
        
        # If steps not provided, select evenly spaced steps
        if steps is None:
            num_steps = min(4, len(self.data['steps']))  # Maximum of 4 distributions
            step_indices = np.linspace(0, len(self.data['steps'])-1, num_steps, dtype=int)
            steps = [self.data['steps'][i] for i in step_indices]
        
        # Create a figure with subplots in a grid
        n_plots = len(steps)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # For each step, plot histogram
        for i, step in enumerate(steps):
            if i >= len(axes):
                break
                
            # Find closest stored step
            step_idx = np.argmin(np.abs(np.array(self.data['steps']) - step))
            actual_step = self.data['steps'][step_idx]
            
            # Get gradient data for this step
            gradients = self.data['gradient_stats'][parameter_name].get(f'step_{actual_step}', None)
            if gradients is None:
                continue
                
            # Plot histogram or KDE
            if isinstance(gradients, list) or isinstance(gradients, np.ndarray):
                axes[i].hist(gradients, bins=30, alpha=0.7, density=True)
                axes[i].set_title(f'Step {actual_step}')
                axes[i].set_xlabel('Gradient Value')
                axes[i].set_ylabel('Density')
            
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Gradient Distribution for {parameter_name.split(".")[-1]}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_convergence_vs_time(self,
                                     save_path: Optional[str] = None,
                                     show: bool = True) -> None:
        """
        Plot objective (loss/accuracy) against wall-clock time.
        
        This visualization shows real-world convergence speed rather than epoch-based.
        
        Args:
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        if 'wall_time' not in self.data:
            raise ValueError("Wall-clock time data not available in the training data")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot loss against wall time
        ax1.plot(self.data['wall_time'], self.data['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in self.data:
            ax1.plot(self.data['wall_time'], self.data['val_loss'], 'g-', label='Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss vs Wall-clock Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy against wall time
        if 'train_acc' in self.data:
            ax2.plot(self.data['wall_time'], self.data['train_acc'], 'b-', label='Train Accuracy')
        if 'val_acc' in self.data:
            ax2.plot(self.data['wall_time'], self.data['val_acc'], 'g-', label='Validation Accuracy')
        ax2.set_xlabel('Wall-clock Time (seconds)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy vs Wall-clock Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_multi_run_comparison(self,
                                      run_results: Dict[str, List[Dict[str, Any]]],
                                      metric: str = 'test_acc',
                                      save_path: Optional[str] = None,
                                      show: bool = True) -> None:
        """
        Create box/violin plots comparing final metrics across multiple runs.
        
        This visualization helps assess optimizer robustness across different seeds.
        
        Args:
            run_results: Dictionary mapping optimizer names to lists of run results
            metric: Metric to compare ('test_acc', 'test_loss', etc.)
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        # Extract final metric values for each optimizer
        optimizer_names = []
        final_metrics = []
        
        for optimizer, runs in run_results.items():
            optimizer_names.append(optimizer)
            metrics_for_optimizer = []
            
            for run in runs:
                if metric in run and len(run[metric]) > 0:
                    # Get the final metric value
                    metrics_for_optimizer.append(run[metric][-1])
            
            final_metrics.append(metrics_for_optimizer)
        
        plt.figure(figsize=(10, 6))
        
        # Create violin plots
        violin_parts = plt.violinplot(final_metrics, showmeans=True)
        
        # Customize violin plots
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(f'C{i}')
            pc.set_alpha(0.7)
        
        plt.xticks(range(1, len(optimizer_names) + 1), optimizer_names)
        plt.ylabel(f'{metric.replace("_", " ").title()}')
        plt.title(f'Distribution of Final {metric.replace("_", " ").title()} Across Multiple Runs')
        plt.grid(True, axis='y')
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_time_to_threshold(self,
                                   run_results: Dict[str, List[Dict[str, Any]]],
                                   metric: str = 'test_acc',
                                   threshold: float = 90.0,
                                   save_path: Optional[str] = None,
                                   show: bool = True) -> None:
        """
        Create box/violin plots comparing time to reach performance threshold.
        
        This visualization helps assess convergence speed reliability across runs.
        
        Args:
            run_results: Dictionary mapping optimizer names to lists of run results
            metric: Metric to use for threshold ('test_acc', 'test_loss', etc.)
            threshold: Performance threshold to measure time to
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        # For loss metrics, we want to find when it goes below the threshold
        # For accuracy metrics, we want to find when it goes above the threshold
        is_loss_metric = 'loss' in metric.lower()
        
        # Extract time to threshold for each optimizer
        optimizer_names = []
        time_to_threshold = []
        
        for optimizer, runs in run_results.items():
            optimizer_names.append(optimizer)
            times_for_optimizer = []
            
            for run in runs:
                if metric in run and 'wall_time' in run:
                    # Find the first step where the metric crosses the threshold
                    for i, value in enumerate(run[metric]):
                        if (is_loss_metric and value <= threshold) or \
                           (not is_loss_metric and value >= threshold):
                            times_for_optimizer.append(run['wall_time'][i])
                            break
            
            time_to_threshold.append(times_for_optimizer)
        
        plt.figure(figsize=(10, 6))
        
        # Create violin plots
        violin_parts = plt.violinplot(time_to_threshold, showmeans=True)
        
        # Customize violin plots
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(f'C{i}')
            pc.set_alpha(0.7)
        
        plt.xticks(range(1, len(optimizer_names) + 1), optimizer_names)
        plt.ylabel('Time to Threshold (seconds)')
        plt.title(f'Distribution of Time to Reach {metric.replace("_", " ").title()} ' +
                 f'{"Below" if is_loss_metric else "Above"} {threshold}')
        plt.grid(True, axis='y')
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def add_multi_run_data(self, 
                          run_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Add results from multiple runs for cross-seed comparison visualizations.
        
        Args:
            run_results: Dictionary mapping optimizer names to lists of run results
        """
        self.multi_run_data = run_results
