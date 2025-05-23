# TALT Optimization Framework Docker Image
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set environment variables for reproducible results
ENV PYTHONHASHSEED=42

# FIXED: Enable CUDA instead of disabling it
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Remove the tokenizers warning
ENV TOKENIZERS_PARALLELISM=false

# Install additional system dependencies not in base image
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt* ./

# Install requirements (PyTorch packages are already in base image)
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy TALT source code
COPY . .

# Create directories for data, results, and configs
RUN mkdir -p /app/data /app/results /app/logs /app/checkpoints

# Set Python path to include both the app directory and talt_evaluation
ENV PYTHONPATH=/app:/app/talt_evaluation

# Create __init__.py files if they don't exist to make directories Python packages
RUN touch /app/__init__.py
RUN if [ ! -f /app/talt_evaluation/__init__.py ]; then touch /app/talt_evaluation/__init__.py; fi

# Default command - show help
CMD ["python", "run_experiments.py", "--help"]

# Labels for metadata
LABEL maintainer="TALT Team"
LABEL description="TALT (Topology-Aware Learning Trajectory) Optimization Framework"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/example/talt"
LABEL org.opencontainers.image.description="Docker image for TALT optimization framework with CUDA support"
LABEL org.opencontainers.image.licenses="MIT"