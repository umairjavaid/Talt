"""Visualization module for the TALT optimizer."""
import os
import logging
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import visualization libraries with fallback handling
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    FuncAnimation = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class TALTVisualizer:
    """
    Unified visualization tools for TALT optimizer.

    Provides methods to visualize:
    1. Loss trajectories and comparison with standard optimizers
    2. Gradient transformations and bifurcation points
    3. Eigenvalue spectra/evolution
    4. Valley detection events
    5. Parameter space projections (if applicable)
    6. Gradient norm history
    7. Animations of loss or other metrics
    """

    def __init__(self, output_dir: str = "./visualizations", max_points: int = 1000):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving visualizations
            max_points: Maximum number of data points to display for dynamic plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_points = max_points
        self.data = {
            'loss_values': deque(maxlen=max_points),
            'valley_detections': [],
            'bifurcations': [],
            'gradient_stats': {},
            'parameter_snapshots': [],
            'eigenvalues_history': {},
            'loss_history_for_landscape': [],
            'grad_memory_for_transformations': {}
        }

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure plotting style
        if MATPLOTLIB_AVAILABLE:
            sns.set_style("darkgrid")
            plt.rcParams.update({
                'figure.figsize': (12, 8),
                'font.size': 12,
                'axes.titlesize': 16,
                'axes.labelsize': 14
            })
        else:
            logger.warning("Matplotlib or Seaborn not found. Visualization capabilities will be limited.")

    def _check_matplotlib(self) -> bool:
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib/Seaborn is required for this visualization function but not installed.")
            return False
        return True
    
    def add_optimizer_data(self, optimizer_data: Dict[str, Any]) -> None:
        """
        Add optimizer data to visualization buffer.
        This method tries to be compatible with data from different TALT versions.

        Args:
            optimizer_data: Dictionary containing optimizer data.
                            Expected keys might include:
                            - 'loss_values': Deque or list of loss values.
                            - 'valley_detections': List of valley detection events/steps.
                            - 'bifurcations': List of bifurcation steps.
                            - 'gradient_stats': Dict of gradient statistics.
                            - 'parameter_snapshots': List of parameter snapshots.
                            - 'eigenvalues': Dict of eigenvalue data (from original TALT).
                            - 'loss_history': List of losses (from original TALT for landscape).
                            - 'grad_memory': Gradient memory (from original TALT).
        """
        if 'loss_values' in optimizer_data:
            if isinstance(optimizer_data['loss_values'], deque):
                self.data['loss_values'].extend(optimizer_data['loss_values'])
            else: # Assuming it's a list, append to deque
                for loss_val in optimizer_data['loss_values']:
                    self.data['loss_values'].append(loss_val)
        
        if 'loss_history' in optimizer_data: # For original_visualizer compatibility
            self.data['loss_history_for_landscape'] = optimizer_data['loss_history']

        if 'valley_detections' in optimizer_data:
            self.data['valley_detections'].extend(optimizer_data['valley_detections'])

        if 'bifurcations' in optimizer_data:
            self.data['bifurcations'].extend(optimizer_data['bifurcations'])

        if 'gradient_stats' in optimizer_data: # From ImprovedTALTVisualizer
            self.data['gradient_stats'].update(optimizer_data['gradient_stats'])
        
        if 'grad_memory' in optimizer_data: # From original_visualizer
            self.data['grad_memory_for_transformations'] = optimizer_data['grad_memory']

        if 'parameter_snapshots' in optimizer_data: # From ImprovedTALTVisualizer
            self.data['parameter_snapshots'].extend(optimizer_data['parameter_snapshots'])
        
        if 'eigenvalues' in optimizer_data: # From original_visualizer (talt_opt._visualization_data['eigenvalues'])
            self.data['eigenvalues_history'] = optimizer_data['eigenvalues']


    # --- Methods from ImprovedTALTVisualizer (adapted) ---
    def visualize_loss_trajectory(self, 
                                save_path: Optional[str] = None, 
                                show: bool = True) -> None:
        """Visualizes the loss trajectory over training steps."""
        if not self._check_matplotlib(): return
        if not self.data['loss_values']:
            logger.info("No loss values available to plot.")
            return

        plt.figure()
        plt.plot(list(self.data['loss_values']), label="Loss")
        
        first_valley_label = True
        for step in self.data['valley_detections']:
            if step < len(self.data['loss_values']):
                plt.axvline(x=step, color='r', linestyle='--', alpha=0.7, label='Valley' if first_valley_label else None)
                first_valley_label = False # Only label first one
        
        first_bifurcation_label = True
        for step in self.data['bifurcations']:
            if step < len(self.data['loss_values']):
                 plt.axvline(x=step, color='g', linestyle=':', alpha=0.7, label='Bifurcation' if first_bifurcation_label else None)
                 first_bifurcation_label = False

        plt.title("Loss Trajectory")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        if self.data['valley_detections'] or self.data['bifurcations'] or self.data['loss_values']:
            plt.legend()
        plt.grid(True, alpha=0.5)
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            logger.info(f"Loss trajectory plot saved to {os.path.join(self.output_dir, save_path)}")
        if show:
            plt.show()
        plt.close()

    def visualize_eigenvalue_spectra(self, 
                                   parameter_name: Optional[str] = None, # Made optional to plot all if None
                                   save_path: Optional[str] = None, 
                                   show: bool = True) -> None:
        """Plots eigenvalue evolution for specified or all parameters."""
        if not self._check_matplotlib(): return
        if not self.data['eigenvalues_history']:
            logger.info("No eigenvalue data available to plot.")
            return

        items_to_plot = self.data['eigenvalues_history'].items()
        if parameter_name:
            if parameter_name not in self.data['eigenvalues_history']:
                logger.warning(f"Eigenvalue data for '{parameter_name}' not found.")
                return
            items_to_plot = [(parameter_name, self.data['eigenvalues_history'][parameter_name])]
        
        num_params_to_plot = len(items_to_plot)
        if num_params_to_plot == 0:
            logger.info("No parameters selected for eigenvalue plot.")
            return

        # Determine number of rows for subplots, max 3 rows
        num_rows = min(3, num_params_to_plot)
        fig, axes = plt.subplots(num_rows, 1, figsize=(12, 4 * num_rows), squeeze=False)
        axes = axes.flatten()
        
        plot_idx = 0
        for (name, eig_data_list) in items_to_plot:
            if plot_idx >= num_rows: break # Stop if we have filled the allocated subplots
            ax = axes[plot_idx]
            if not eig_data_list: continue
            
            steps = [item[0] for item in eig_data_list]
            eigenvalues_over_time = [item[1] for item in eig_data_list]
            
            # Plotting top k eigenvalues, e.g., top 5
            k = min(5, len(eigenvalues_over_time[0]) if eigenvalues_over_time and eigenvalues_over_time[0] is not None else 0)
            if k == 0: continue

            for i in range(k):
                # Ensure eigenvalue arrays are not empty and access them safely
                current_eigenvalues = [eigs[i] if eigs is not None and i < len(eigs) else np.nan for eigs in eigenvalues_over_time]
                ax.plot(steps, current_eigenvalues, label=f'Eig {i+1}')
            
            ax.set_title(f"Top {k} Eigenvalues for {name}")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Eigenvalue")
            ax.legend()
            plot_idx += 1
            
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            logger.info(f"Eigenvalue spectra plot saved to {os.path.join(self.output_dir, save_path)}")
        if show:
            plt.show()
        plt.close()


    def visualize_gradient_norm_history(self, 
                                      save_path: Optional[str] = None, 
                                      show: bool = True) -> None:
        """Visualizes the history of gradient norms for different parameter groups."""
        if not self._check_matplotlib(): return
        if not self.data['gradient_stats'] or not any(self.data['gradient_stats'].values()):
            logger.info("No gradient norm data available to plot.")
            return

        plt.figure()
        for param_name, norms in self.data['gradient_stats'].items():
            if norms: # Ensure there's data to plot
                plt.plot(norms, label=param_name)
        
        plt.title("Gradient Norm History")
        plt.xlabel("Training Step / Update")
        plt.ylabel("Gradient Norm")
        if any(self.data['gradient_stats'].values()): plt.legend()
        plt.grid(True, alpha=0.5)

        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            logger.info(f"Gradient norm history plot saved to {os.path.join(self.output_dir, save_path)}")
        if show:
            plt.show()
        plt.close()

    # --- Methods from TALTVisualizer (original_visualizer.py, adapted) ---
    def plot_optimizer_comparison(self, 
                                std_res: Dict[str, List[float]], 
                                talt_res: Dict[str, List[float]],
                                save_path: Optional[str] = None,
                                show: bool = True) -> None:
        """Plot comparison between standard and TALT optimizers."""
        if not self._check_matplotlib(): return
        plt.figure(figsize=(12, 8))
        
        metrics = [('train_loss', 'Train Loss'), ('test_loss', 'Test Loss'), 
                   ('train_acc', 'Train Accuracy'), ('test_acc', 'Test Accuracy')]
        
        plotted_something = False
        for i, (key, title) in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            has_std = key in std_res and std_res[key]
            has_talt = key in talt_res and talt_res[key]
            if has_std:
                plt.plot(std_res[key], label=f'Standard - {title.split()[0]}') # Shorter label
                plotted_something = True
            if has_talt:
                plt.plot(talt_res[key], label=f'TALT - {title.split()[0]}') # Shorter label
                plotted_something = True
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(title.split()[-1]) # Loss or Accuracy
            if has_std or has_talt: plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            logger.info(f"Optimizer comparison plot saved to {os.path.join(self.output_dir, save_path)}")
        if show:
            plt.show()
        plt.close()
        
    def visualize_gradient_transformations(self,
                                         save_path: Optional[str] = None,
                                         show: bool = True) -> None:
        """Visualize gradient transformations over time."""
        if not self._check_matplotlib(): return
        
        # Updated to use correct data key
        grad_memory = self.data.get('grad_memory_for_transformations', self.data.get('grad_memory', {}))
        
        if not grad_memory:
            logger.info("No gradient transformation data available")
            return
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Gradient Transformations Analysis")
        
        # Plot gradient norms for different parameters
        param_items = list(grad_memory.items())[:4]  # Limit to 4 parameters
        
        for i, (param_name, grad_data) in enumerate(param_items):
            ax = axes[i // 2, i % 2]
            
            if isinstance(grad_data, list) and grad_data:
                # Handle different grad_data formats
                if isinstance(grad_data[0], tuple) and len(grad_data[0]) >= 2:
                    # Format: [(step, norm, ...)]
                    steps, norms, _ = zip(*grad_data)
                elif hasattr(grad_data[0], 'norm'):
                    # Tensor objects
                    steps = list(range(len(grad_data)))
                    norms = [g.norm().item() if hasattr(g, 'norm') else float(g) for g in grad_data]
                else:
                    # Simple list of values
                    steps = list(range(len(grad_data)))
                    norms = grad_data
                
                ax.plot(steps, norms, label=f'{param_name} grad norm')
                ax.set_title(f"Gradient Norm: {param_name}")
                ax.set_xlabel("Step")
                ax.set_ylabel("Norm")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Gradient Norm: {param_name}")
        
        plt.tight_layout()
        if save_path:
            full_path = os.path.join(self.output_dir, save_path) if not os.path.isabs(save_path) else save_path
            plt.savefig(full_path, dpi=150, bbox_inches='tight')
            logger.info(f"Gradient transformations plot saved to {full_path}")
        if show:
            plt.show()
        plt.close()

    def visualize_loss_landscape_with_valleys(self,
                                            save_path: Optional[str] = None,
                                            show: bool = True) -> None:
        """Visualize loss landscape with valley/bifurcation detections."""
        if not self._check_matplotlib(): return
        
        # Updated to use correct data keys
        loss_history = self.data.get('loss_history_for_landscape', self.data.get('loss_history', list(self.data['loss_values'])))
        detection_points = self.data.get('valley_detections', [])
        
        if not loss_history:
            logger.info("No loss history to plot")
            return
            
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', label='Loss', linewidth=1.5)
        
        # Mark valley detections
        first_detection_labeled = False
        for step in detection_points:
            if step < len(loss_history):
                label = 'Valley Detection' if not first_detection_labeled else None
                plt.axvline(x=step, color='red', linestyle='--', alpha=0.7, label=label)
                first_detection_labeled = True
                
        plt.title("Loss Landscape with Valley/Bifurcation Detections")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        if first_detection_labeled or len(loss_history) > 0:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            full_path = os.path.join(self.output_dir, save_path) if not os.path.isabs(save_path) else save_path
            plt.savefig(full_path, dpi=150, bbox_inches='tight')
            logger.info(f"Loss landscape plot saved to {full_path}")
        if show:
            plt.show()
        plt.close()

    # --- Animation and Report (from ImprovedTALTVisualizer, can be kept as is or adapted) ---
    def create_animation(self, data_key: str = 'loss_values', fps: int = 5, 
                         save_path: str = "talt_animation.mp4") -> None:
        """
        Creates an animation of a specified metric over time.
        Requires ffmpeg to be installed and on PATH.
        """
        if not self._check_matplotlib() or FuncAnimation is None:
            logger.error("Matplotlib FuncAnimation is required for animation but not available/imported.")
            return
        
        if data_key not in self.data or not self.data[data_key]:
            logger.info(f"No data available for animation key: {data_key}")
            return
            
        fig, ax = plt.subplots()
        data_to_animate = list(self.data[data_key])
        if not data_to_animate:
            logger.info(f"Empty data for animation key: {data_key}")
            return
            
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, len(data_to_animate) -1 if len(data_to_animate) > 1 else 1)
        
        min_val = min(data_to_animate)
        max_val = max(data_to_animate)
        padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
        ax.set_ylim(min_val - padding, max_val + padding)

        ax.set_title(f"{data_key.replace('_', ' ').title()} Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        def animate(frame):
            x_data = list(range(frame + 1))
            y_data = data_to_animate[:frame + 1]
            line.set_data(x_data, y_data)
            return line,

        try:
            anim = FuncAnimation(fig, animate, frames=len(data_to_animate), 
                               interval=1000//fps, blit=True, repeat=False)
            
            full_save_path = os.path.join(self.output_dir, save_path)
            anim.save(full_save_path, writer='ffmpeg', fps=fps)
            logger.info(f"Animation saved to {full_save_path}")
            
        except Exception as e:
            logger.error(f"Failed to create animation: {e}")
        finally:
            plt.close(fig)

    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> None:
        """Generate a comprehensive visualization report with all available plots."""
        if not self._check_matplotlib():
            return
            
        report_dir = self.output_dir if save_path is None else Path(save_path).parent
        report_name = "talt_comprehensive_report" if save_path is None else Path(save_path).stem
        
        # Generate all individual visualizations
        try:
            self.visualize_loss_trajectory(
                save_path=f"{report_name}_loss_trajectory.png",
                show=False
            )
            self.visualize_eigenvalue_spectra(
                save_path=f"{report_name}_eigenvalue_spectra.png", 
                show=False
            )
            self.visualize_gradient_norm_history(
                save_path=f"{report_name}_gradient_norms.png",
                show=False
            )
            self.visualize_gradient_transformations(
                save_path=f"{report_name}_gradient_transformations.png",
                show=False
            )
            self.visualize_loss_landscape_with_valleys(
                save_path=f"{report_name}_loss_landscape.png",
                show=False
            )
            
            logger.info(f"Comprehensive TALT visualization report generated in {report_dir}")
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
