# TALT Optimizer Evaluation Framework

This framework provides a comprehensive testing environment to evaluate the performance of the Topology-Aware Learning Trajectory (TALT) optimizer across different neural network architectures and datasets.

## Overview

The TALT Evaluation Framework allows systematic comparison between TALT and standard optimizers (SGD, Adam) across:

- **CNN Architectures**: ResNet (18, 50), VGG16, EfficientNet-B0, SimpleCNN
- **LLM Architecture**: BERT-base
- **Datasets**: CIFAR10, CIFAR100, GLUE SST-2, MNIST

## Features

- **Hyperparameter tuning** of TALT optimizer using Optuna
- **Visualization tools** for training metrics, feature maps, and attention patterns
- **Reproducible experiment configurations** via JSON
- **Batch processing** for running multiple experiments
- **Mixed-precision training** support
- **Checkpoint management** for saving and resuming training

## Directory Structure

```
talt_evaluation/
├── architectures/        # Architecture implementations
│   ├── cnn/              # CNN model implementations
│   └── llm/              # BERT implementation
├── batch_configs/        # JSON configuration files for batch experiments
├── datasets/             # Dataset loaders for CIFAR and GLUE
├── experiments/          # Experiment framework
├── hyperparameter_tuning/ # Hyperparameter optimization module
├── visualization/        # Visualization tools
├── run_experiment.py     # Script to run single experiment
└── run_batch.py          # Script to run experiment batches
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/talt-evaluation.git
   cd talt-evaluation
   ```

2. Install required packages:
   ```bash
   pip install torch torchvision optuna matplotlib seaborn transformers datasets
   ```

3. Install the TALT optimizer:
   ```bash
   pip install talt
   ```

## Usage

### Running a Single Experiment

To run a single experiment:

```bash
python run_experiment.py --name resnet50_cifar100_talt \
                         --architecture resnet50 \
                         --dataset cifar100 \
                         --optimizer talt \
                         --epochs 30 \
                         --batch-size 128
```

### Running BERT on GLUE SST-2

```bash
python run_experiment.py --name bert_sst2_talt \
                         --architecture bert-base \
                         --dataset glue-sst2 \
                         --optimizer talt \
                         --epochs 5 \
                         --batch-size 32 \
                         --lr 2e-5
```

### Tuning TALT Hyperparameters

```bash
python run_experiment.py --name resnet50_cifar100_tuning \
                         --architecture resnet50 \
                         --dataset cifar100 \
                         --optimizer talt \
                         --tune-hyperparams \
                         --n-trials 30
```

### Running a Batch of Experiments

```bash
python run_batch.py --config batch_configs/cnn_comparison.json \
                    --output-dir ./results \
                    --gpu-indices 0,1 \
                    --parallel
```

## Batch Configuration

You can define batch experiments in JSON files. For example:

```json
{
  "description": "Comparison of TALT vs SGD on ResNet models",
  "experiments": [
    {
      "name": "resnet18_cifar10_talt",
      "architecture": "resnet18",
      "dataset": "cifar10",
      "optimizer": "talt",
      "epochs": 30,
      "batch-size": 128,
      "lr": 0.1
    },
    {
      "name": "resnet18_cifar10_sgd",
      "architecture": "resnet18",
      "dataset": "cifar10",
      "optimizer": "sgd",
      "epochs": 30,
      "batch-size": 128,
      "lr": 0.1
    }
  ]
}
```

## Hyperparameter Tuning

The framework includes an Optuna-based hyperparameter tuning module for TALT. Tunable parameters include:

- `learning_rate`
- `projection_dim`
- `memory_size`
- `update_interval`
- `valley_strength`
- `smoothing_factor`
- `grad_store_interval`
- `cov_decay`
- `adaptive_reg`

## Results and Visualizations

After running experiments, the framework generates:

1. **Training metrics**: Loss and accuracy curves
2. **Architecture-specific visualizations**:
   - CNN: Feature maps from different layers
   - BERT: Attention maps and token importance
3. **TALT optimizer analysis**: Visualization of the optimization trajectory
4. **Comparative reports**: Performance comparison between optimizers

Results are saved in the specified output directory with the following structure:

```
results/
├── experiment_name_timestamp/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── checkpoint_epoch_N.pt
│   ├── visualizations/
│   │   ├── learning_curves.png
│   │   ├── feature_maps.png
│   │   └── attention_maps.png
│   ├── config.json
│   └── results.json
└── batch_name_timestamp/
    ├── experiment_1/
    ├── experiment_2/
    └── batch_summary.json
```

## Extending the Framework

### Adding New Architectures

1. Create a new model class inheriting from `BaseArchitecture`
2. Implement the required methods:
   - `get_optimizer_config()`
   - `get_hyperparameter_search_space()`
   - `architecture_specific_visualization()`

### Adding New Datasets

1. Create a new dataset loader in the `datasets` directory
2. Update `get_dataset()` in `datasets/__init__.py`

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- Optuna
- matplotlib
- seaborn
- transformers
- datasets
- TALT optimizer