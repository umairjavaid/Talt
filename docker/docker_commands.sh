#!/bin/bash

# Optimized Docker commands for TALT framework with better PyTorch performance

# Color variables for output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Optimized Docker run flags for PyTorch
DOCKER_GPU_FLAGS="--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=2g"

# Function to build the Docker image
build_image() {
    echo -e "${YELLOW}Building TALT Docker image...${NC}"
    
    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Build from the parent directory (where the source code is)
    BUILD_CONTEXT="$(dirname "$SCRIPT_DIR")"
    
    # Build the image
    docker build -t talt:latest -f "$SCRIPT_DIR/Dockerfile" "$BUILD_CONTEXT"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Docker image built successfully!${NC}"
        echo -e "${GREEN}Image tagged as: talt:latest${NC}"
    else
        echo -e "${RED}❌ Docker build failed!${NC}"
        exit 1
    fi
}

# Function to run single experiment with optimized settings
run_experiment() {
    echo -e "${YELLOW}Running TALT experiment with optimized settings...${NC}"
    docker run --rm ${DOCKER_GPU_FLAGS} \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/data:/app/data \
        talt:latest python run_experiments.py single "$@"
}

# Function to run batch experiments with optimized settings
run_batch() {
    echo -e "${YELLOW}Running batch experiments with optimized settings...${NC}"
    docker run --rm ${DOCKER_GPU_FLAGS} \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/data:/app/data \
        talt:latest python run_experiments.py batch "$@"
}

# Function to start interactive shell with optimized settings
start_shell() {
    echo -e "${YELLOW}Starting interactive shell with GPU support...${NC}"
    docker run --rm -it ${DOCKER_GPU_FLAGS} \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/data:/app/data \
        talt:latest bash
}

# Test CUDA with optimized settings
test_cuda() {
    echo -e "${YELLOW}Testing CUDA with optimized Docker settings...${NC}"
    docker run --rm ${DOCKER_GPU_FLAGS} talt:latest python -c "
import torch
print('=== CUDA Test Results ===')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    # Test tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print('✅ GPU tensor operations successful')
print('=========================')"
}

# Quick experiment test
quick_test() {
    echo -e "${YELLOW}Running quick TALT test...${NC}"
    docker run --rm ${DOCKER_GPU_FLAGS} \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/data:/app/data \
        talt:latest python run_experiments.py single \
        --architecture simplecnn --dataset cifar10 --optimizer talt \
        --epochs 1 --batch-size 32
}

# Usage examples
usage() {
    echo "Optimized TALT Docker Commands:"
    echo ""
    echo "  ./docker_commands.sh build          - Build the Docker image"
    echo "  ./docker_commands.sh test_cuda      - Test CUDA setup"
    echo "  ./docker_commands.sh quick_test     - Quick TALT test"
    echo "  ./docker_commands.sh shell          - Interactive shell"
    echo ""
    echo "Standard commands (now optimized):"
    echo "  ./docker_commands.sh run --architecture resnet18 --dataset cifar10 --optimizer talt"
    echo "  ./docker_commands.sh batch --config batch_configs/cnn_comparison.json"
}

case "$1" in
    build)
        build_image
        ;;
    test_cuda)
        test_cuda
        ;;
    quick_test)
        quick_test
        ;;
    shell)
        start_shell
        ;;
    run)
        shift
        run_experiment "$@"
        ;;
    batch)
        shift
        run_batch "$@"
        ;;
    *)
        usage
        ;;
esac