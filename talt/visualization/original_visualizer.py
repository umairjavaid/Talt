"""Visualization module for the original TALT optimizer."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class OriginalTALTVisualizer:
    """
    Visualization tool for the Original TALT optimizer.
    
    This class provides methods to visualize:
    1. Loss trajectory
    2. Eigenvalue spectra
    3. Valley detection points
    4. Gradient magnitude changes
    5. Topology changes
    """
    
    def __init__(self, 
                 output_dir: str = './visualizations',
                 experiment_name: str = 'talt_original',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        """
        Initialize the TALT visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            experiment_name: Name of the experiment
            figsize: Figure size for plots
            dpi: DPI for saved figures
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.figsize = figsize
        self.dpi = dpi
        
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_optimizer_state(self, optimizer_data: Dict[str, Any]):
        """
        Create visualizations based on optimizer state.
        
        Args:
            optimizer_data: Dictionary containing optimizer state data
                - loss_values: List of loss values
                - bifurcation_points: List of valley detection points
                - eigenvalues: Dict mapping parameter names to eigenvalue data
                - grad_memory: Dict mapping parameter names to gradient data
        """
        # Check if the required data is available
        if not optimizer_data:
            logger.warning("No optimizer data provided for visualization")
            return []
        
        generated_files = []
        
        # Visualize loss trajectory and valleys
        if 'loss_values' in optimizer_data and optimizer_data['loss_values']:
            loss_file = self._visualize_loss_trajectory(
                optimizer_data['loss_values'],
                optimizer_data.get('bifurcation_points', [])
            )
            generated_files.append(loss_file)
        
        # Visualize eigenvalue spectra for a subset of parameters
        if 'eigenvalues' in optimizer_data and optimizer_data['eigenvalues']:
            eigen_files = self._visualize_eigenvalue_spectra(optimizer_data['eigenvalues'])
            generated_files.extend(eigen_files)
        
        # Visualize gradient magnitudes
        if 'grad_memory' in optimizer_data and optimizer_data['grad_memory']:
            grad_files = self._visualize_gradient_magnitudes(optimizer_data['grad_memory'])
            generated_files.extend(grad_files)
        
        return generated_files
    
    def _visualize_loss_trajectory(self, 
                                  loss_values: List[float], 
                                  bifurcation_points: List[int]) -> str:
        """
        Visualize the loss trajectory with valley detection points.
        
        Args:
            loss_values: List of loss values
            bifurcation_points: List of iteration indices where valleys were detected
            
        Returns:
            str: Path to the saved figure
        """
        plt.figure(figsize=self.figsize)
        
        # Plot loss trajectory
        plt.plot(loss_values, label='Loss', color='blue', alpha=0.7)
        
        # Mark bifurcation points (valleys)
        if bifurcation_points:
            for bp in bifurcation_points:
                if 0 <= bp < len(loss_values):
                    plt.axvline(x=bp, color='red', linestyle='--', alpha=0.5)
        
        plt.title('Loss Trajectory with Valley Detection Points')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Save figure
        output_path = os.path.join(self.output_dir, f"{self.experiment_name}_loss_trajectory.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _visualize_eigenvalue_spectra(self, 
                                    eigenvalue_data: Dict[str, List[Tuple[int, List[float]]]]) -> List[str]:
        """
        Visualize eigenvalue spectra for selected parameters.
        
        Args:
            eigenvalue_data: Dictionary mapping parameter names to eigenvalue data
                Each entry is a list of (step, eigenvalues) tuples
                
        Returns:
            List[str]: Paths to the saved figures
        """
        saved_files = []
        
        # Select a subset of parameters (at most 5) to visualize
        param_names = list(eigenvalue_data.keys())
        if len(param_names) > 5:
            param_names = param_names[:5]
        
        for param_name in param_names:
            eigenvalues_history = eigenvalue_data[param_name]
            
            if not eigenvalues_history:
                continue
            
            plt.figure(figsize=self.figsize)
            
            # Get the eigenvalues at different steps
            steps = []
            top_eigenvalues = []  # Track top 3 eigenvalues
            
            for step, eigenvals in eigenvalues_history:
                steps.append(step)
                
                # If eigenvalues is a tensor, convert to numpy
                if isinstance(eigenvals, torch.Tensor):
                    eigenvals = eigenvals.cpu().numpy()
                
                # Take top 3 eigenvalues
                top_n = min(3, len(eigenvals))
                top_eigenvalues.append(eigenvals[:top_n])
            
            # Plot top eigenvalues
            if top_eigenvalues:
                top_eigenvalues = np.array(top_eigenvalues)
                for i in range(top_eigenvalues.shape[1]):
                    plt.plot(steps, top_eigenvalues[:, i], 
                             label=f'Eigenvalue {i+1}', 
                             alpha=0.7)
            
            plt.title(f'Top Eigenvalues for {param_name.split(".")[-1]}')
            plt.xlabel('Step')
            plt.ylabel('Eigenvalue Magnitude')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Save figure
            param_short_name = param_name.split('.')[-1]
            output_path = os.path.join(self.output_dir, f"{self.experiment_name}_{param_short_name}_eigenvalues.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            saved_files.append(output_path)
        
        return saved_files
    
    def _visualize_gradient_magnitudes(self, 
                                     grad_data: Dict[str, List[Tuple[int, float, float]]]) -> List[str]:
        """
        Visualize gradient magnitude changes for selected parameters.
        
        Args:
            grad_data: Dictionary mapping parameter names to gradient data
                Each entry is a list of (step, grad_norm, transformed_norm) tuples
                
        Returns:
            List[str]: Paths to the saved figures
        """
        saved_files = []
        
        # Select a subset of parameters (at most 5) to visualize
        param_names = list(grad_data.keys())
        if len(param_names) > 5:
            param_names = param_names[:5]
        
        for param_name in param_names:
            grad_history = grad_data[param_name]
            
            if not grad_history:
                continue
            
            plt.figure(figsize=self.figsize)
            
            # Extract gradient data
            steps = []
            grad_norms = []
            
            for data_point in grad_history:
                if len(data_point) >= 2:
                    step, grad_norm = data_point[0], data_point[1]
                    steps.append(step)
                    grad_norms.append(grad_norm)
            
            plt.plot(steps, grad_norms, label='Gradient Norm', alpha=0.7)
            
            plt.title(f'Gradient Magnitude for {param_name.split(".")[-1]}')
            plt.xlabel('Step')
            plt.ylabel('Gradient Norm')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Save figure
            param_short_name = param_name.split('.')[-1]
            output_path = os.path.join(self.output_dir, f"{self.experiment_name}_{param_short_name}_gradient.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            saved_files.append(output_path)
        
        return saved_files
    
    def generate_comprehensive_report(self, optimizer_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive HTML report with all visualizations.
        
        Args:
            optimizer_data: Dictionary containing optimizer state data
                
        Returns:
            str: Path to the generated HTML report
        """
        # Generate all visualizations
        viz_files = self.visualize_optimizer_state(optimizer_data)
        
        # Create a simple HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Original TALT Visualization Report: {self.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333366; }}
                .viz-section {{ margin-bottom: 30px; }}
                .viz-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .viz-item {{ margin-bottom: 20px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Original TALT Visualization Report: {self.experiment_name}</h1>
            
            <div class="viz-section">
                <h2>Visualizations</h2>
                <div class="viz-container">
        """
        
        # Add visualization images
        for viz_file in viz_files:
            file_name = os.path.basename(viz_file)
            html_content += f"""
                    <div class="viz-item">
                        <h3>{file_name}</h3>
                        <img src="{viz_file}" alt="{file_name}">
                    </div>
            """
        
        # Close HTML tags
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = os.path.join(self.output_dir, f"{self.experiment_name}_visualization_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
