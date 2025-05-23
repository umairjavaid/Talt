#!/bin/bash

# Common Docker commands for TALT framework

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}TALT Docker Management Script${NC}"

# Function to print usage
usage() {
    echo "Usage: $0 {build|run|batch|comprehensive|shell|clean|logs}"
    echo ""
    echo "Commands:"
    echo "  build         - Build the TALT Docker image"
    echo "  run           - Run a single experiment"
    echo "  batch         - Run batch experiments"
    echo "  comprehensive - Run comprehensive evaluation"
    echo "  shell         - Start interactive shell in container"
    echo "  clean         - Clean up Docker images and containers"
    echo "  logs          - Show container logs"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run --architecture resnet18 --dataset cifar10 --optimizer talt"
    echo "  $0 batch --config talt_evaluation/batch_configs/cnn_comparison.json"
}

# Function to build Docker image
build_image() {
    echo -e "${YELLOW}Building TALT Docker image...${NC}"
    docker build -f docker/Dockerfile -t talt:latest .
    echo -e "${GREEN}Build completed!${NC}"
}

# Function to run single experiment
run_experiment() {
    echo -e "${YELLOW}Running TALT experiment...${NC}"
    docker run --rm --gpus all \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/data:/app/data \
        talt:latest python run_experiments.py single "$@"
}

# Function to run batch experiments
run_batch() {
    echo -e "${YELLOW}Running batch experiments...${NC}"
    docker run --rm --gpus all \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/data:/app/data \
        talt:latest python run_experiments.py batch "$@"
}

# Function to run comprehensive evaluation
run_comprehensive() {
    echo -e "${YELLOW}Running comprehensive evaluation...${NC}"
    docker run --rm --gpus all \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/data:/app/data \
        talt:latest python run_comprehensive_talt_evaluation.py "$@"
}

# Function to start interactive shell
start_shell() {
    echo -e "${YELLOW}Starting interactive shell...${NC}"
    docker run --rm -it --gpus all \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/data:/app/data \
        talt:latest bash
}

# Function to clean up
cleanup() {
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    docker system prune -f
    docker image prune -f
    echo -e "${GREEN}Cleanup completed!${NC}"
}

# Function to show logs
show_logs() {
    echo -e "${YELLOW}Showing container logs...${NC}"
    docker logs talt-framework 2>/dev/null || echo "No running container found"
}

# Create necessary directories
mkdir -p results data logs checkpoints

# Main command handling
case "$1" in
    build)
        build_image
        ;;
    run)
        shift
        run_experiment "$@"
        ;;
    batch)
        shift
        run_batch "$@"
        ;;
    comprehensive)
        shift
        run_comprehensive "$@"
        ;;
    shell)
        start_shell
        ;;
    clean)
        cleanup
        ;;
    logs)
        show_logs
        ;;
    *)
        usage
        exit 1
        ;;
esac
