# TALT Optimization Framework

TALT is a novel optimization framework for training neural networks.

## Quick Start

### Option 1: Local Installation

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

### Option 2: Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Talt.git
cd Talt
```

2. Build the Docker image:
```bash
./scripts/docker_commands.sh build
```

3. Run a single experiment:
```bash
./scripts/docker_commands.sh run --architecture resnet18 --dataset cifar10 --optimizer talt
```

## Docker Usage

TALT provides Docker support for easy setup and reproducible experiments. All commands are managed through the `docker_commands.sh` script.

### Building the Docker Image

```bash
./scripts/docker_commands.sh build
```

### Running Experiments

#### Single Experiment
```bash
./scripts/docker_commands.sh run --architecture resnet18 --dataset cifar10 --optimizer talt
```

#### Batch Experiments
```bash
./scripts/docker_commands.sh batch --config talt_evaluation/batch_configs/cnn_comparison.json
```

#### Comprehensive Evaluation
```bash
./scripts/docker_commands.sh comprehensive --configs cnn_comparison
```

### Interactive Analysis

Start Jupyter notebook for interactive analysis and visualization:
```bash
./scripts/docker_commands.sh jupyter
```

This will start a Jupyter server accessible at `http://localhost:8888` for interactive development and result analysis.

## Usage

### Single Experiment

#### Local
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

#### Docker
```bash
./scripts/docker_commands.sh run \
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

#### Local
```bash
python run_experiments.py batch \
    --config talt_evaluation/batch_configs/cnn_comparison.json \
    --output-dir ./results \
    --gpu-indices 0,1 \
    --parallel
```

#### Docker
```bash
./scripts/docker_commands.sh batch \
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