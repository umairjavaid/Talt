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
                       include_animations: bool = False) -> None:
        """
        Generate a comprehensive visualization report.

        Args:
            experiment_name: Name of the experiment
            include_animations: Whether to include animations
        """
        # Create experiment directory
        exp_dir = os.path.join(self.output_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"Generating visualization report in {exp_dir}...")
        
        # Generate loss trajectory visualization
        self.visualize_loss_trajectory(
            save_path=os.path.join(exp_dir, "loss_trajectory.png"),
            show=False
        )
        
        # Generate eigenvalue visualizations for each parameter
        for param_name in self.data['gradient_stats'].keys():
            short_name = param_name.split('.')[-1]
            self.visualize_eigenvalue_spectra(
                parameter_name=param_name,
                save_path=os.path.join(exp_dir, f"eigenvalues_{short_name}.png"),
                show=False
            )
        
        # Generate gradient norm history
        self.visualize_gradient_norm_history(
            save_path=os.path.join(exp_dir, "gradient_norms.png"),
            show=False
        )
        
        # Generate animations if requested
        if include_animations:
            self.create_animation(
                save_path=os.path.join(exp_dir, "loss_animation.mp4")
            )
            
        print(f"Visualization report generated in {exp_dir}")
