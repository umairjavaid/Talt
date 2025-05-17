# Topology-Aware Learning Trajectory (TALT) Optimizer

An enhanced optimizer for deep learning that uses topology analysis to improve optimization performance and convergence.

## Features

- **Dimensionality Reduction**: Uses sparse random projection to efficiently process high-dimensional gradients
- **Incremental Covariance Estimation**: Memory-efficient tracking of gradient statistics
- **Robust Eigendecomposition**: Power iteration method for stable eigenvalue computation
- **Non-parametric Valley Detection**: Identifies valleys in the loss landscape for faster convergence
- **Visualization Tools**: Comprehensive visualization of optimization trajectories and landscape features

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/talt.git
cd talt
pip install -e .
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

Use the provided example script to run experiments:

```bash
# Train with standard SGD
python -m examples.main --dataset cifar10 --epochs 10

# Train with improved TALT optimizer
python -m examples.main --dataset cifar10 --epochs 10 --use-talt
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