#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer

def plot_attention_head(attention, tokens, head_idx, layer_idx, ax=None):
    """
    Plot a single attention head as a heatmap.
    
    Args:
        attention: Attention weights [seq_len, seq_len]
        tokens: List of token strings
        head_idx: Index of the attention head
        layer_idx: Index of the transformer layer
        ax: Matplotlib axis for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create attention heatmap
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=np.max(attention),
        ax=ax
    )
    
    ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    return ax

def plot_attention_heads_grid(attention_data, tokens, layer_indices=None, head_indices=None, output_path=None):
    """
    Plot multiple attention heads in a grid.
    
    Args:
        attention_data: Dictionary of attention weights {layer_idx: tensor}
        tokens: List of token strings
        layer_indices: List of layer indices to plot, or None for all
        head_indices: List of head indices to plot, or None for all
        output_path: Path to save the plot
    """
    # If not specified, use all layers
    if layer_indices is None:
        layer_indices = sorted([int(k.split('_')[1]) for k in attention_data.keys()])
    
    # Get number of heads from first layer
    first_layer_key = f"layer_{layer_indices[0]}"
    num_heads = attention_data[first_layer_key].shape[1]
    
    # If not specified, use all heads
    if head_indices is None:
        head_indices = list(range(num_heads))
    
    # Create a grid of subplots
    n_layers = len(layer_indices)
    n_heads = len(head_indices)
    
    # Limit the number of plots if there are too many
    if n_layers * n_heads > 12:
        # Select a subset of layers and heads
        step_layers = max(1, n_layers // 3)
        step_heads = max(1, n_heads // 4)
        layer_indices = layer_indices[::step_layers]
        head_indices = head_indices[::step_heads]
        n_layers = len(layer_indices)
        n_heads = len(head_indices)
    
    # Calculate grid dimensions
    n_cols = min(n_heads, 4)
    n_rows = (n_layers * n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    
    # If there's only one plot, axes is not an array
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    plot_idx = 0
    for layer_idx in layer_indices:
        layer_key = f"layer_{layer_idx}"
        if layer_key not in attention_data:
            continue
        
        layer_attention = attention_data[layer_key][0]  # First example in batch
        
        for head_idx in head_indices:
            if head_idx >= layer_attention.shape[0]:
                continue
            
            if plot_idx >= len(axes):
                break
            
            head_attention = layer_attention[head_idx].numpy()
            plot_attention_head(head_attention, tokens, head_idx, layer_idx, axes[plot_idx])
            plot_idx += 1
    
    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_attention_by_token(attention_data, tokens, token_idx, output_path=None):
    """
    Plot attention focused on a specific token across layers.
    
    Args:
        attention_data: Dictionary of attention weights {layer_idx: tensor}
        tokens: List of token strings
        token_idx: Index of the token to focus on
        output_path: Path to save the plot
    """
    layer_indices = sorted([int(k.split('_')[1]) for k in attention_data.keys()])
    n_layers = len(layer_indices)
    
    fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 5, 5))
    
    # If there's only one layer, axes is not an array
    if n_layers == 1:
        axes = [axes]
    
    focused_token = tokens[token_idx] if token_idx < len(tokens) else "UNK"
    plt.suptitle(f"Attention focused on token: '{focused_token}'", fontsize=16)
    
    for i, layer_idx in enumerate(layer_indices):
        layer_key = f"layer_{layer_idx}"
        if layer_key not in attention_data:
            continue
        
        layer_attention = attention_data[layer_key][0]  # First example
        
        # Average attention across all heads
        avg_attention = layer_attention.mean(0).numpy()
        
        # Get attention to the selected token
        token_attention = avg_attention[:, token_idx]
        
        # Plot as a horizontal bar chart
        axes[i].barh(range(len(tokens)), token_attention, color='skyblue')
        axes[i].set_yticks(range(len(tokens)))
        axes[i].set_yticklabels(tokens)
        axes[i].set_xlabel('Attention Weight')
        axes[i].set_title(f'Layer {layer_idx+1}')
        axes[i].invert_yaxis()  # Tokens read top to bottom
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_bert_attention(vis_data, output_path_prefix):
    """
    Create visualizations for BERT attention maps.
    
    Args:
        vis_data: Visualization data from BERT model
        output_path_prefix: Prefix for output paths
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
    
    attention_maps = vis_data.get('attention_maps', {})
    if not attention_maps:
        print("No attention maps available for visualization")
        return
    
    # Initialize tokenizer to decode tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # For demonstration, we'll use the first example in the batch
    input_ids = vis_data.get('input_ids', None)
    
    # If input IDs are not provided in vis_data, generate a dummy list of tokens
    if input_ids is not None:
        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    else:
        tokens = [f"Token{i}" for i in range(attention_maps[list(attention_maps.keys())[0]].shape[2])]
    
    # Plot attention head grid for selected layers
    layers_to_plot = [0, 3, 7, 11]  # First, middle, and last layers
    layers_to_plot = [l for l in layers_to_plot if f"layer_{l}" in attention_maps]
    
    if layers_to_plot:
        plot_attention_heads_grid(
            attention_maps,
            tokens,
            layer_indices=layers_to_plot,
            output_path=f"{output_path_prefix}_heads_grid.png"
        )
    
    # Plot attention focused on [CLS] token
    plot_attention_by_token(
        attention_maps,
        tokens,
        token_idx=0,  # [CLS] token is usually first
        output_path=f"{output_path_prefix}_cls_focus.png"
    )
    
    # Plot attention focused on the last token (often [SEP])
    if len(tokens) > 1:
        plot_attention_by_token(
            attention_maps,
            tokens,
            token_idx=len(tokens) - 1,
            output_path=f"{output_path_prefix}_sep_focus.png"
        )