#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_feature_map(feature_map, ax=None, title=None):
    """
    Plot a single feature map.
    
    Args:
        feature_map: 2D tensor or array representing a feature map
        ax: Matplotlib axis for plotting
        title: Title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.cpu().numpy()
    
    # Normalize feature map values for better visualization
    norm = Normalize(vmin=feature_map.min(), vmax=feature_map.max())
    
    # Plot feature map as an image
    im = ax.imshow(feature_map, cmap='viridis', norm=norm)
    
    if title:
        ax.set_title(title)
    
    ax.axis('off')
    return ax

def plot_multiple_feature_maps(feature_maps, n_maps=16, output_path=None):
    """
    Plot multiple feature maps from a convolutional layer.
    
    Args:
        feature_maps: 3D tensor or array of shape [channels, height, width]
        n_maps: Number of feature maps to plot
        output_path: Path to save the plot
    """
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.cpu().numpy()
    
    # Get number of available channels
    n_channels = feature_maps.shape[0]
    n_maps = min(n_maps, n_channels)
    
    # Calculate grid dimensions
    n_cols = min(4, n_maps)
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    # Handle the case where there's only one row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Sample feature maps evenly throughout the channels
    if n_maps < n_channels:
        indices = np.linspace(0, n_channels - 1, n_maps, dtype=int)
    else:
        indices = np.arange(n_maps)
    
    for i, idx in enumerate(indices):
        r, c = divmod(i, n_cols)
        plot_feature_map(feature_maps[idx], ax=axes[r, c], title=f'Channel {idx}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_layer_feature_maps(activations, output_dir, n_maps_per_layer=9):
    """
    Plot feature maps from multiple layers.
    
    Args:
        activations: Dictionary mapping layer names to activation tensors
        output_dir: Directory to save the plots
        n_maps_per_layer: Number of feature maps to plot per layer
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, activation in activations.items():
        output_path = os.path.join(output_dir, f"{layer_name}_feature_maps.png")
        
        # For each activation tensor, plot feature maps
        plot_multiple_feature_maps(
            activation, 
            n_maps=n_maps_per_layer, 
            output_path=output_path
        )

def plot_original_images(images, output_path=None):
    """
    Plot original input images.
    
    Args:
        images: Batch of images as tensor or array [batch, channels, height, width]
        output_path: Path to save the plot
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    
    batch_size = min(4, images.shape[0])
    
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
    
    # Handle the case where there's only one image
    if batch_size == 1:
        axes = [axes]
    
    for i in range(batch_size):
        img = images[i]
        
        # For grayscale images
        if img.shape[0] == 1:
            img = img[0]
            axes[i].imshow(img, cmap='gray')
        # For RGB images
        elif img.shape[0] == 3:
            # Transpose from [channels, height, width] to [height, width, channels]
            img = np.transpose(img, (1, 2, 0))
            
            # Normalize to [0, 1] for visualization
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            axes[i].imshow(img)
        
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_cnn_feature_maps(vis_data, output_path_prefix):
    """
    Create visualizations for CNN feature maps.
    
    Args:
        vis_data: Visualization data from CNN model
        output_path_prefix: Prefix for output paths
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
    
    # Get feature maps from the visualization data
    feature_maps = vis_data.get('feature_maps', {})
    
    if not feature_maps:
        print("No feature maps available for visualization")
        return
    
    # Plot feature maps for each layer
    for layer_name, maps in feature_maps.items():
        output_path = f"{output_path_prefix}_{layer_name}.png"
        plot_multiple_feature_maps(maps, n_maps=16, output_path=output_path)
    
    # Create a summary visualization showing one feature map from each layer
    fig, axes = plt.subplots(1, len(feature_maps), figsize=(len(feature_maps) * 4, 4))
    
    # Handle the case where there's only one layer
    if len(feature_maps) == 1:
        axes = [axes]
    
    for i, (layer_name, maps) in enumerate(feature_maps.items()):
        # Show mean feature map
        mean_map = maps.mean(dim=0) if isinstance(maps, torch.Tensor) else maps.mean(axis=0)
        plot_feature_map(mean_map, ax=axes[i], title=layer_name)
    
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_summary.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # If original input images are provided in the visualization data
    if 'input_images' in vis_data:
        plot_original_images(vis_data['input_images'], 
                             output_path=f"{output_path_prefix}_input_images.png")