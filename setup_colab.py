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
    4. Creates symlinks to ensure consistent paths
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
    
    # Add the current directory to Python path
    sys.path.insert(0, str(root_dir))
    print(f"Added {root_dir} to Python path")
    
    # Create __init__.py files to make modules importable
    modules = ['talt_evaluation', 
               'talt_evaluation/datasets', 
               'talt_evaluation/architectures', 
               'talt_evaluation/hyperparameter_tuning',
               'talt_evaluation/visualization',
               'talt_evaluation/experiments']
    
    for module in modules:
        module_dir = root_dir / module
        module_dir.mkdir(exist_ok=True)
        init_file = module_dir / '__init__.py'
        if not init_file.exists():
            with open(init_file, 'w') as f:
                pass
            print(f"Created {init_file}")
    
    # Create a symlink to prevent path issues in Colab
    source_dir = root_dir / "talt_evaluation"
    if source_dir.exists():
        print("Ensuring module structure is correct...")
        # Make symlinks for key modules
        for module in ['datasets', 'architectures', 'experiments', 
                       'hyperparameter_tuning', 'visualization']:
            module_path = source_dir / module
            if not module_path.exists():
                module_path.mkdir(exist_ok=True)
                print(f"Created directory: {module_path}")
            
            # Create __init__.py if it doesn't exist
            init_file = module_path / '__init__.py'
            if not init_file.exists():
                with open(init_file, 'w') as f:
                    pass
                print(f"Created {init_file}")
    
    print("Setup completed successfully!")
    print("\nYou can now run experiments with:")
    print("python talt_evaluation/run_batch.py --config talt_evaluation/batch_configs/cnn_comparison.json --output-dir ./results")

if __name__ == "__main__":
    setup_colab()