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

# Install core ML and scientific computing dependencies
# Note: PyTorch is already installed in the base image
RUN pip install \
    transformers>=4.21.0 \
    datasets>=2.0.0 \
    huggingface-hub>=0.10.0 \
    optuna>=3.0.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pandas>=1.4.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.1.0 \
    scipy>=1.8.0 \
    tqdm>=4.64.0 \
    pillow>=9.0.0 \
    opencv-python>=4.5.0 \
    plotly>=5.0.0 \
    kaleido>=0.2.1

# Install Jupyter and notebook dependencies
RUN pip install \
    jupyter>=1.0.0 \
    jupyterlab>=3.4.0 \
    notebook>=6.4.0 \
    ipywidgets>=7.7.0 \
    nbconvert>=6.5.0

# Install additional utilities
RUN pip install \
    psutil>=5.9.0 \
    gpustat>=1.0.0 \
    tensorboard>=2.9.0 \
    wandb>=0.13.0

# Create non-root user for security
RUN useradd -m -u 1000 talt && \
    mkdir -p /home/talt && \
    chown -R talt:talt /home/talt

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt* ./

# Install any additional requirements if they exist
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy TALT source code
COPY --chown=talt:talt . .

# Install TALT package in editable mode
RUN pip install -e .

# Create directories for data, results, and configs
RUN mkdir -p /app/data /app/results /app/logs /app/checkpoints && \
    chown -R talt:talt /app

# Create Jupyter config directory
RUN mkdir -p /home/talt/.jupyter && \
    chown -R talt:talt /home/talt/.jupyter

# Generate Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/talt/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/talt/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /home/talt/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /home/talt/.jupyter/jupyter_notebook_config.py && \
    chown -R talt:talt /home/talt/.jupyter

# Switch to non-root user
USER talt

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check to verify installation
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import torch; import talt; print('TALT installation verified'); \
                   print(f'PyTorch version: {torch.__version__}'); \
                   print(f'CUDA available: {torch.cuda.is_available()}'); \
                   print(f'CUDA devices: {torch.cuda.device_count()}')" || exit 1

# Expose Jupyter port
EXPOSE 8888

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# TALT Docker Entrypoint\n\
set -e\n\
\n\
# Print system information\n\
echo "=== TALT Framework Docker Container ==="\n\
echo "Python version: $(python --version)"\n\
echo "PyTorch version: $(python -c \"import torch; print(torch.__version__)\")"\n\
echo "CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\")"\n\
echo "CUDA devices: $(python -c \"import torch; print(torch.cuda.device_count())\")"\n\
echo "Working directory: $(pwd)"\n\
echo "User: $(whoami)"\n\
echo "======================================="\n\
\n\
# Set CUDA device if specified\n\
if [ ! -z "$CUDA_DEVICE" ]; then\n\
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"\n\
    echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"\n\
fi\n\
\n\
# Execute the command\n\
exec "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - show help
CMD ["python", "run_experiments.py", "--help"]

# Labels for metadata
LABEL maintainer="TALT Team"
LABEL description="TALT (Topology-Aware Learning Trajectory) Optimization Framework"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/example/talt"
LABEL org.opencontainers.image.description="Docker image for TALT optimization framework with CUDA support"
LABEL org.opencontainers.image.licenses="MIT"
