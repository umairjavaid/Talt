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
COPY ../requirements.txt* ./

# Install requirements (PyTorch packages are already in base image)
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy TALT source code
COPY .. .

# Install TALT package in editable mode
RUN pip install -e .

# Create directories for data, results, and configs
RUN mkdir -p /app/data /app/results /app/logs /app/checkpoints

# Create Jupyter config directory
RUN mkdir -p /root/.jupyter

# Generate Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py

# Set Python path
ENV PYTHONPATH=/app

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
