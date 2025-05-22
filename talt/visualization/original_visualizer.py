"""Visualization module for the original TALT optimizer."""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class TALTVisualizer:
    """Visualization utilities for the original TALT optimizer."""
    
    @staticmethod
    def plot_results(std_res: Dict[str, List[float]], talt_res: Dict[str, List[float]]) -> None:
        """Plot comparison between standard and TALT optimizers."""
        plt.figure(figsize=(12, 8))
        titles = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]
        
        for i, (k1, k2) in enumerate([("train_loss", "train_acc"), ("test_loss", "test_acc")], 1):
            plt.subplot(2, 2, i * 2 - 1)
            plt.plot(std_res[k1], label="Standard")
            plt.plot(talt_res[k1], label="TALT")
            plt.title(titles[(i - 1) * 2])
            plt.xlabel("Epoch")
            plt.ylabel("Loss" if "loss" in k1 else "Accuracy (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, i * 2)
            plt.plot(std_res[k2], label="Standard")
            plt.plot(talt_res[k2], label="TALT")
            plt.title(titles[(i - 1) * 2 + 1])
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)" if "acc" in k2 else "Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def visualize_gradient_transformations(talt_opt) -> None:
        """Visualize gradient transformations with bifurcation points."""
        weight_params = [n for n in talt_opt.grad_memory if ".weight" in n]
        bias_params = [n for n in talt_opt.grad_memory if ".bias" in n]
        
        if not weight_params and not bias_params:
            logger.warning("No gradient data available")
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        for ax, params, label in zip(axes, (weight_params, bias_params), ("Weight", "Bias")):
            if not params:
                ax.set_visible(False)
                continue
                
            grad_data = []
            for name in params:
                if name in talt_opt.grad_memory and talt_opt.grad_memory[name]:
                    grads = [g.cpu().numpy() for g in talt_opt.grad_memory[name]]
                    if grads:
                        grad_data.append(np.vstack(grads))
            
            if not grad_data:
                ax.set_visible(False)
                continue
                
            all_grads = np.vstack(grad_data)
            mean = all_grads.mean(0)
            std = all_grads.std(0)
            
            x = np.arange(len(mean))
            ax.plot(x, mean, label=f"{label} Gradient Mean")
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            
            # Mark bifurcation points (valley detections)
            for step in talt_opt.bifurcations:
                if step < len(x):
                    ax.axvline(step, color="g", linestyle="--", alpha=0.3)
                    
            ax.set_title(f"{label} Gradient Analysis")
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Magnitude")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def analyze_eigenvalues(talt_opt) -> None:
        """Plot eigenvalue evolution over training."""
        if not talt_opt._visualization_data['eigenvalues']:
            logger.warning("No eigenvalue data available")
            return
            
        fig, axes = plt.subplots(min(3, len(talt_opt._visualization_data['eigenvalues'])), 
                                 1, figsize=(12, 10))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
            
        for (name, eig_data), ax in zip(talt_opt._visualization_data['eigenvalues'].items(), axes):
            if not eig_data:
                ax.set_visible(False)
                continue
                
            eig_data = np.array(eig_data)
            for i in range(min(3, eig_data.shape[1])):
                ax.plot(eig_data[:, i], label=f"λ{i+1}")
                
            ax.set_title(f"Top Eigenvalues: {name}")
            ax.set_xlabel("Update Step")
            ax.set_ylabel("Eigenvalue")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_loss_landscape(talt_opt) -> None:
        """Plot loss with valley detection markers."""
        if not talt_opt.loss_history:
            logger.warning("No loss history available")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(talt_opt.loss_history, label="Loss")
        
        for step in talt_opt.bifurcations:
            if step < len(talt_opt.loss_history):
                plt.axvline(step, color="r", linestyle="--", alpha=0.5, 
                           label="Valley Detection" if step == talt_opt.bifurcations[0] else "")
                
        plt.title("Loss Landscape with Valley Detections")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
