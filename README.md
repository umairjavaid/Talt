# TALT: Trajectory-Aware Learning Technique

## Overview
TALT is a novel optimization technique for training neural networks that analyzes the optimization trajectory to improve convergence.

## Features

- **Dimensionality Reduction**: Uses sparse random projection to efficiently process high-dimensional gradients
- **Incremental Covariance Estimation**: Memory-efficient tracking of gradient statistics
- **Robust Eigendecomposition**: Power iteration method for stable eigenvalue computation
- **Non-parametric Valley Detection**: Identifies valleys in the loss landscape for faster convergence
- **Visualization Tools**: Comprehensive visualization of optimization trajectories and landscape features

## Setup

### Local Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Talt.git
cd Talt

# Install the package
pip install -e .
```

### Google Colab Setup
If you're using Google Colab, run the following commands:

```python
!git clone https://github.com/yourusername/Talt.git
%cd Talt
!python setup_colab.py
```

## Quick Start

```python
import torch
import torch.nn as nn
import talt

# Define model
model = talt.SimpleCNN(num_channels=3, image_size=32, num_classes=10)

# Load data
train_loader, test_loader, _, _, _ = talt.get_loaders(dataset_name='cifar10', batch_size=128)

# Create optimizer
optimizer = talt.ImprovedTALTOptimizer(
    model=model,
    base_optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
    lr=0.01,
    projection_dim=32,
    update_interval=20,
    valley_strength=0.2
)

# Train model
results = talt.train_and_evaluate_improved(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=10,
    use_improved_talt=True
)
```

## Running Experiments

### Single Experiment
To run a single experiment:

```bash
python talt_evaluation/run_experiment.py \
    --name resnet18_cifar10_talt \
    --architecture resnet18 \
    --dataset cifar10 \
    --optimizer talt \
    --epochs 30 \
    --batch-size 128 \
    --lr 0.1 \
    --mixed-precision \
    --output-dir ./results
```

### Batch Experiments
To run a batch of experiments:

```bash
python talt_evaluation/run_batch.py \
    --config talt_evaluation/batch_configs/cnn_comparison.json \
    --output-dir ./results
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to use (cifar10, mnist) | cifar10 |
| `--batch-size` | Training batch size | 128 |
| `--epochs` | Number of training epochs | 10 |
| `--lr` | Initial learning rate | 0.01 |
| `--use-talt` | Enable improved TALT optimizer | False |
| `--projection-dim` | Dimension after projection | 32 |
| `--update-interval` | Steps between topology updates | 20 |
| `--valley-strength` | Valley acceleration strength | 0.2 |
| `--smoothing-factor` | Curvature smoothing factor | 0.3 |

## Configuration
Experiment configurations are defined in JSON files under `talt_evaluation/batch_configs/`.

## Package Structure

```
talt/
├── __init__.py              # Package exports
├── components/              # Core components
│   ├── __init__.py
│   ├── covariance.py        # Incremental covariance estimation
│   ├── dimensionality_reduction.py  # Random projection
│   ├── eigendecomposition.py  # Power iteration method
│   └── valley_detection.py  # Valley detection algorithm
├── model/                   # Neural network models
│   ├── __init__.py
│   └── cnn.py               # Simple CNN implementation
├── optimizer/               # Optimizer implementation
│   ├── __init__.py
│   └── improved_talt.py     # Main optimizer implementation
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── monitoring.py        # Performance monitoring
├── visualization/           # Visualization tools
│   ├── __init__.py
│   └── visualizer.py        # Visualization components
└── train.py                 # Training utilities
```

## How It Works

The Improved TALT Optimizer enhances traditional optimizers in several ways:

1. **Gradient Transformation**: Analyzes gradient statistics to create optimal transformations
2. **Valley Detection**: Identifies and accelerates across valleys in the loss landscape
3. **Adaptive Regularization**: Automatically adjusts regularization based on landscape condition
4. **Memory Efficiency**: Uses dimensionality reduction to minimize memory footprint

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.