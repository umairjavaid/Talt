"""Visualization module for the TALT optimizer."""
from typing import Dict, List, Any, Optional
from collections import deque
import os
import logging
# Attempt to import plotting libraries, and provide a fallback or warning if not available.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Mock objects or functions if needed for the class to be defined without errors
    class plt_mock:
        @staticmethod
        def figure(*args, **kwargs): pass
        @staticmethod
        def plot(*args, **kwargs): pass
        @staticmethod
        def axvline(*args, **kwargs): pass
        @staticmethod
        def scatter(*args, **kwargs): pass
        @staticmethod
        def title(*args, **kwargs): pass
        @staticmethod
        def xlabel(*args, **kwargs): pass
        @staticmethod
        def ylabel(*args, **kwargs): pass
        @staticmethod
        def legend(*args, **kwargs): pass
        @staticmethod
        def grid(*args, **kwargs): pass
        @staticmethod
        def savefig(*args, **kwargs): pass
        @staticmethod
        def show(*args, **kwargs): pass
        @staticmethod
        def close(*args, **kwargs): pass
        @staticmethod
        def subplots(*args, **kwargs): return (None, [plt_mock()] * 4) # return a mock figure and axes
        @staticmethod
        def tight_layout(*args, **kwargs): pass
        @staticmethod
        def rcParams(*args, **kwargs): return type('MockDict', (dict,), {'update': lambda s, d: None})()


    class sns_mock:
        @staticmethod
        def set_style(*args, **kwargs): pass

    plt = plt_mock
    sns = sns_mock
    FuncAnimation = None # Will be checked before use

import numpy as np

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
        self.output_dir = output_dir
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
                self.data['loss_values'] = optimizer_data['loss_values']
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
        """Visualize gradient transformations with bifurcation points, using stored grad_memory."""
        if not self._check_matplotlib(): return
        grad_memory = self.data.get('grad_memory_for_transformations', {})
        if not grad_memory:
            logger.info("No gradient memory data available for visualizing transformations.")
            return

        weight_params = [n for n in grad_memory if ".weight" in n and grad_memory[n]]
        bias_params = [n for n in grad_memory if ".bias" in n and grad_memory[n]]
        
        if not weight_params and not bias_params:
            logger.info("No weight or bias parameters with stored gradients found.")
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), squeeze=False) # Ensure axes is 2D
        axes = axes.flatten()

        param_groups = [(weight_params, "Weight Gradients"), (bias_params, "Bias Gradients")]
        first_bif_label_grad = True

        for ax_idx, (params_to_plot, label_prefix) in enumerate(param_groups):
            ax = axes[ax_idx]
            if not params_to_plot:
                ax.text(0.5, 0.5, f"No {label_prefix.lower().split()[0]} data", ha='center', va='center')
                ax.set_title(f"{label_prefix} (No Data)")
                continue
            
            plotted_on_ax = False
            for name in params_to_plot:
                # grad_memory[name] is a list of (step, grad_projection_norm, original_grad_norm)
                if not grad_memory[name]: continue # Skip if no data for this param
                steps = [item[0] for item in grad_memory[name]]
                transformed_norms = [item[1] for item in grad_memory[name]]
                original_norms = [item[2] for item in grad_memory[name]]

                ax.plot(steps, original_norms, '--', alpha=0.7, label=f'{name} (Orig.)')
                ax.plot(steps, transformed_norms, label=f'{name} (Trans.)')
                plotted_on_ax = True

            for bif_step in self.data.get('bifurcations', []):
                ax.axvline(x=bif_step, color='purple', linestyle=':', linewidth=2, label='Bifurcation' if first_bif_label_grad else None)
                first_bif_label_grad = False

            ax.set_title(f"{label_prefix} - Norms Before and After Transformation")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Gradient Norm")
            if plotted_on_ax or self.data.get('bifurcations', []): ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            logger.info(f"Gradient transformations plot saved to {os.path.join(self.output_dir, save_path)}")
        if show:
            plt.show()
        plt.close()
    
    def visualize_loss_landscape_with_valleys(self,
                                            save_path: Optional[str] = None,
                                            show: bool = True) -> None:
        """Plot loss with valley detection markers, using stored loss history and bifurcations."""
        if not self._check_matplotlib(): return
        loss_history = self.data.get('loss_history_for_landscape', [])
        if not loss_history: # Fallback to loss_values if specific history isn't there
            loss_history = list(self.data['loss_values'])
        
        if not loss_history:
            logger.info("No loss history available for landscape visualization.")
            return
            
        plt.figure() # New figure
        plt.plot(loss_history, label="Loss")
        
        # Use bifurcations as proxy for valley detections if valley_detections is empty
        detection_points = self.data.get('valley_detections', [])
        detection_label_prefix = "Valley"
        if not detection_points: # If no valley_detections, use bifurcations
            detection_points = self.data.get('bifurcations', [])
            detection_label_prefix = "Bifurcation"

        first_detection_labeled = True
        for step in detection_points:
            if step < len(loss_history):
                label = None
                if first_detection_labeled:
                    label = f'{detection_label_prefix} Detected'
                    first_detection_labeled = False
                plt.axvline(x=step, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=label)
                plt.scatter(step, loss_history[step], color='red', marker='o', s=50, zorder=5) # Mark point on loss curve
                
        plt.title("Loss Landscape with Valley/Bifurcation Detections")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        if first_detection_labeled == False or len(loss_history) > 0:
             plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            logger.info(f"Loss landscape plot saved to {os.path.join(self.output_dir, save_path)}")
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
            logger.warning(f"No data found for key '{data_key}' to create animation.")
            return

        fig, ax = plt.subplots()
        data_to_animate = list(self.data[data_key])
        if not data_to_animate: # Extra check for empty list after conversion
            logger.warning(f"Data for key '{data_key}' is empty. Cannot create animation.")
            plt.close(fig)
            return
            
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, len(data_to_animate) -1 if len(data_to_animate) > 1 else 1) # Adjust xlim for single point
        
        min_val = min(data_to_animate)
        max_val = max(data_to_animate)
        padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1 # Handle case where all values are same
        ax.set_ylim(min_val - padding, max_val + padding)

        ax.set_title(f"{data_key.replace('_', ' ').title()} Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel(data_key.replace('_', ' ').title())

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            x = list(range(i + 1))
            y = data_to_animate[:i + 1]
            line.set_data(x, y)
            return line,

        try:
            anim = FuncAnimation(fig, animate, init_func=init,
                                 frames=len(data_to_animate), interval=max(1, 1000//fps), blit=True)
            
            animation_save_path = os.path.join(self.output_dir, save_path)
            anim.save(animation_save_path, writer='ffmpeg', fps=fps)
            logger.info(f"Animation saved to {animation_save_path}")
        except Exception as e:
            logger.error(f"Failed to create animation: {e}. Ensure ffmpeg is installed and in PATH.")
        finally:
            plt.close(fig)

    def generate_report(self, experiment_name: str = "TALT_Experiment", 
                        include_plots: Optional[List[str]] = None) -> None:
        """
        Generates a simple HTML report with key visualizations.
        'include_plots' can specify which plots to generate and embed, e.g.,
        ['loss_trajectory', 'eigenvalue_spectra', 'gradient_transformations']
        """
        if not self._check_matplotlib():
            logger.error("Cannot generate report without Matplotlib/Seaborn.")
            return
            
        if include_plots is None:
            include_plots = ['loss_trajectory', 'loss_landscape_with_valleys', 'gradient_transformations', 'eigenvalue_spectra']

        report_path = os.path.join(self.output_dir, f"{experiment_name}_report.html")
        
        with open(report_path, 'w') as f:
            f.write(f"<html><head><title>{experiment_name} Report</title></head><body>")
            f.write(f"<h1>Report for {experiment_name}</h1>")

            # Generate and embed plots
            if 'loss_trajectory' in include_plots:
                loss_traj_path = f"{experiment_name}_loss_trajectory.png"
                self.visualize_loss_trajectory(save_path=loss_traj_path, show=False)
                if os.path.exists(os.path.join(self.output_dir, loss_traj_path)):
                    f.write(f"<h2>Loss Trajectory</h2><img src='{loss_traj_path}' alt='Loss Trajectory'><br>")

            if 'loss_landscape_with_valleys' in include_plots:
                loss_land_path = f"{experiment_name}_loss_landscape.png"
                self.visualize_loss_landscape_with_valleys(save_path=loss_land_path, show=False)
                if os.path.exists(os.path.join(self.output_dir, loss_land_path)):
                    f.write(f"<h2>Loss Landscape with Detections</h2><img src='{loss_land_path}' alt='Loss Landscape'><br>")
            
            if 'gradient_transformations' in include_plots:
                grad_trans_path = f"{experiment_name}_grad_transformations.png"
                self.visualize_gradient_transformations(save_path=grad_trans_path, show=False)
                if os.path.exists(os.path.join(self.output_dir, grad_trans_path)):
                    f.write(f"<h2>Gradient Transformations</h2><img src='{grad_trans_path}' alt='Gradient Transformations'><br>")

            if 'eigenvalue_spectra' in include_plots and self.data['eigenvalues_history']:
                 # Plot for the first parameter group with eigenvalue data, or all if fewer than 3.
                param_to_plot = next(iter(self.data['eigenvalues_history']), None)
                if param_to_plot:
                    eig_spec_path = f"{experiment_name}_eigenvalue_spectra.png"
                    self.visualize_eigenvalue_spectra(parameter_name=None, save_path=eig_spec_path, show=False) # Plot all (up to 3 subplots)
                    if os.path.exists(os.path.join(self.output_dir, eig_spec_path)):
                        f.write(f"<h2>Eigenvalue Spectra</h2><img src='{eig_spec_path}' alt='Eigenvalue Spectra'><br>")
            
            f.write("</body></html>")
        logger.info(f"Report generated at {report_path}")
        # Optionally, clean up individual plot files if they are only for the report
        # for plot_file in plot_files.values():
        #     try:
        #         os.remove(os.path.join(self.output_dir, plot_file))
        #     except OSError as e:
        #         logger.warning(f"Could not remove plot file {plot_file}: {e}")

# Example usage (for testing this file directly):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    vis = TALTVisualizer(output_dir="./talt_visualizations_output")

    if MATPLOTLIB_AVAILABLE:
        optimizer_data_example = {
            'loss_values': deque([10, 8, 6, 5, 4.5, 4, 3.8, 3.5, 3.4, 3.3] * 3, maxlen=vis.max_points),
            'valley_detections': [5, 15],
            'bifurcations': [7, 17],
            'loss_history': [10, 8, 6, 5, 4.5, 4, 3.8, 3.5, 3.4, 3.3] * 3,
            'grad_memory': {
                'layer1.weight': [(i*2, np.random.rand()*0.5, np.random.rand()) for i in range(10)],
                'layer1.bias': [(i*2, np.random.rand()*0.2, np.random.rand()*0.5) for i in range(10)]
            },
            'eigenvalues': { 
                'layer1.weight': [(i*2, np.sort(np.random.rand(5))[::-1]*10) for i in range(10)],
                'layer2.weight': [(i*2, np.sort(np.random.rand(5))[::-1]*5) for i in range(10)]
            },
            'gradient_stats': {
                'fc.weight': [np.random.rand() for _ in range(30)],
                'conv1.weight': [np.random.rand()*0.5 for _ in range(30)]
            }
        }
        vis.add_optimizer_data(optimizer_data_example)

        vis.visualize_loss_trajectory(save_path="loss_traj.png", show=False)
        vis.visualize_loss_landscape_with_valleys(save_path="loss_landscape.png", show=False)
        vis.visualize_gradient_transformations(save_path="grad_transform.png", show=False)
        vis.visualize_eigenvalue_spectra(save_path="eigen_spectra.png", show=False)
        vis.visualize_gradient_norm_history(save_path="grad_norm_hist.png", show=False)

        std_data = {'train_loss': [1, 0.8, 0.6, 0.5, 0.4], 'test_acc': [0.5, 0.6, 0.7, 0.72, 0.75]}
        talt_data = {'train_loss': [0.9, 0.7, 0.5, 0.4, 0.3], 'test_acc': [0.55, 0.65, 0.75, 0.77, 0.8]}
        vis.plot_optimizer_comparison(std_data, talt_data, save_path="optimizer_comp.png", show=False)
        
        vis.create_animation(data_key='loss_values', save_path="loss_animation.mp4")
        vis.generate_report(experiment_name="Test_TALT_Run")

        print(f"All visualizations saved in {vis.output_dir}")
    else:
        print("Matplotlib/Seaborn not available, skipping direct visualization tests.")
        # You can still test data addition and other non-plotting logic
        vis.add_optimizer_data({'loss_values': deque([1,2,3])})
        print(f"Data added: {vis.data['loss_values']}")
