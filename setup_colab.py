#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from pathlib import Path

def setup_colab():
    """
    Set up the TALT environment for Google Colab.
    This script:
    1. Installs the required packages
    2. Sets up the Python path
    3. Creates necessary directories
    """
    print("Setting up TALT in Google Colab environment...")
    
    # Check if we're running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
        
    if not IN_COLAB:
        print("Not running in Google Colab. No setup needed.")
        return
    
    # Get the root directory of the project
    root_dir = Path(os.getcwd())
    
    # Install the package in development mode
    print("Installing talt package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
    
    # Create results directory if it doesn't exist
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Created results directory: {results_dir}")
    
    print("Setup completed successfully!")
    print("\nYou can now run experiments with:")
    print("python talt_evaluation/run_batch.py --config talt_evaluation/batch_configs/cnn_comparison.json --output-dir ./results")

if __name__ == "__main__":
    setup_colab()