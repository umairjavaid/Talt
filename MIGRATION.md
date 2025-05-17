# Migration Guide: Moving from Notebook to Package Structure

This guide explains how to migrate from the original notebook implementation to the new refactored TALT package.

## Original Implementation vs. Refactored Package

The original implementation was contained in a single notebook, which combined several components:

1. Timer and performance monitoring classes
2. Neural network model definition
3. Dimensionality reduction and covariance estimation
4. Eigendecomposition
5. Valley detection logic
6. The main optimizer implementation
7. Visualization tools
8. Training and evaluation functions

The refactored package separates these components into a proper Python package structure:

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

## Migration Map

Here's how classes and functions from the original notebook map to the new package:

| Original Component | New Location |
|--------------------|-------------|
| `Timer` | `talt.utils.monitoring.Timer` |
| `print_memory_usage` | `talt.utils.monitoring.print_memory_usage` |
| `PerformanceTracker` | `talt.utils.monitoring.PerformanceTracker` |
| `RandomProjection` | `talt.components.dimensionality_reduction.RandomProjection` |
| `IncrementalCovariance` | `talt.components.covariance.IncrementalCovariance` |
| `PowerIteration` | `talt.components.eigendecomposition.PowerIteration` |
| `ValleyDetector` | `talt.components.valley_detection.ValleyDetector` |
| `SimpleCNN` | `talt.model.cnn.SimpleCNN` |
| `ImprovedTALTOptimizer` | `talt.optimizer.improved_talt.ImprovedTALTOptimizer` |
| `ImprovedTALTVisualizer` | `talt.visualization.visualizer.ImprovedTALTVisualizer` |
| Training functions | `talt.train.train_and_evaluate_improved` |

## How to Migrate Your Code

### From:

```python
# Original notebook code
import torch

# Define classes inline
class Timer:
    # ...class implementation...

class ImprovedTALTOptimizer:
    # ...optimizer implementation...

# Create model and optimizer
model = SimpleCNN(num_channels=3, image_size=32, num_classes=10)
optimizer = ImprovedTALTOptimizer(model, torch.optim.SGD, lr=0.01)

# Train the model
train_and_evaluate(model, optimizer, train_loader, test_loader)
```

### To:

```python
# New refactored package
import torch
import talt

# Create model and optimizer
model = talt.SimpleCNN(num_channels=3, image_size=32, num_classes=10)
optimizer = talt.ImprovedTALTOptimizer(
    model=model,
    base_optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr),
    lr=0.01
)

# Train the model
train_loader, test_loader, _, _, _ = talt.get_loaders('cifar10', batch_size=128)
results = talt.train_and_evaluate_improved(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    use_improved_talt=True
)
```

## Example Usage

See the provided example notebook (`examples/talt_demo.ipynb`) and script (`examples/main.py`) for
detailed usage examples and code patterns.

## Benefits of the Refactored Structure

1. **Modularity**: Components can be tested, updated, and extended independently
2. **Reusability**: Functions and classes can be imported and reused in other projects
3. **Maintainability**: Better organization makes the code easier to understand and maintain
4. **Documentation**: Better documentation and examples make the code more approachable
5. **Packaging**: The code can be installed as a Python package using pip
