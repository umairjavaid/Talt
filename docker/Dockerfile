# TALT Optimization Framework Docker Image
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set environment variables for reproducible results
ENV PYTHONHASHSEED=42
ENV CUDA_VISIBLE_DEVICES=""
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

# Install TALT package in editable mode
RUN pip install -e .

# Create directories for data, results, and configs
RUN mkdir -p /app/data /app/results /app/logs /app/checkpoints

# Set Python path
ENV PYTHONPATH=/app

# Default command - show help
CMD ["python", "run_experiments.py", "--help"]

# Labels for metadata
LABEL maintainer="TALT Team"
LABEL description="TALT (Topology-Aware Learning Trajectory) Optimization Framework"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/example/talt"
LABEL org.opencontainers.image.description="Docker image for TALT optimization framework with CUDA support"
LABEL org.opencontainers.image.licenses="MIT"
