#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run TALT experiments in Google Colab')
    
    parser.add_argument('--config', type=str, default='talt_evaluation/batch_configs/cnn_comparison.json',
                        help='Path to batch configuration JSON file')
    parser.add_argument('--output-dir', type=str, default='./results', 
                        help='Base output directory for all experiments')
    
    return parser.parse_args()

def main():
    """
    Main entry point for running TALT experiments in Google Colab.
    This script:
    1. Sets up the environment
    2. Ensures path correctness
    3. Runs the experiments
    """
    print("Setting up TALT environment in Google Colab...")
    
    # Get the current directory (should be the project root)
    root_dir = Path(os.getcwd())
    
    # First, run the setup script
    setup_script = root_dir / "setup_colab.py"
    if setup_script.exists():
        print("Running setup script...")
        subprocess.run([sys.executable, str(setup_script)])
    else:
        print(f"Warning: Setup script not found at {setup_script}")
        
    # Parse arguments
    args = parse_args()
    
    # Ensure the config file exists
    config_path = root_dir / args.config
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
        
    # Run the batch script with the absolute path to ensure consistency
    batch_script = root_dir / "talt_evaluation" / "run_batch.py"
    if not batch_script.exists():
        print(f"Error: Batch script not found at {batch_script}")
        return
        
    print(f"Running experiments with config: {args.config}")
    
    # Run from the root directory
    os.chdir(root_dir)
    
    # Construct and run the command
    cmd = [
        sys.executable,
        str(batch_script),
        "--config", args.config,
        "--output-dir", args.output_dir
    ]
    
    print(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=root_dir)
    
if __name__ == "__main__":
    main()