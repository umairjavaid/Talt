# TALT Optimization Framework

TALT is a novel optimization framework for training neural networks.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Talt.git
cd Talt
```

2. Install the package and dependencies:
```bash
pip install -e .
```

3. Run a single experiment:
```bash
python run_experiments.py single --architecture resnet18 --dataset cifar10 --optimizer talt
```

4. Run a batch of experiments:
```bash
python run_experiments.py batch --config talt_evaluation/batch_configs/cnn_comparison.json
```

## Usage

### Single Experiment

```bash
python run_experiments.py single \
    --name "my_experiment" \
    --architecture resnet18 \
    --dataset cifar10 \
    --optimizer talt \
    --epochs 30 \
    --batch-size 128 \
    --lr 0.1 \
    --mixed-precision \
    --save-checkpoints \
    --output-dir ./results
```

### Batch Experiments

```bash
python run_experiments.py batch \
    --config talt_evaluation/batch_configs/cnn_comparison.json \
    --output-dir ./results \
    --gpu-indices 0,1 \
    --parallel
```

## Configuration Files

Batch configuration files are JSON files specifying parameters for multiple experiments.
Example (`talt_evaluation/batch_configs/cnn_comparison.json`):

```json
{
  "experiments": [
    {
      "name": "resnet18_cifar10_talt",
      "architecture": "resnet18",
      "dataset": "cifar10",
      "optimizer": "talt",
      "epochs": 30,
      "batch-size": 128,
      "lr": 0.1,
      "mixed-precision": true,
      "save-checkpoints": true
    },
    {
      "name": "resnet18_cifar10_sgd",
      "architecture": "resnet18",
      "dataset": "cifar10",
      "optimizer": "sgd",
      "epochs": 30,
      "batch-size": 128,
      "lr": 0.1,
      "mixed-precision": true,
      "save-checkpoints": true
    }
  ]
}
```