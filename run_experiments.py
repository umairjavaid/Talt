#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for running TALT experiments.
This script provides a simple interface to run both single experiments and batch experiments.

Example usage:
    # Run a single experiment
    python run_experiments.py single --architecture resnet18 --dataset cifar10 --optimizer talt

    # Run a batch of experiments
    python run_experiments.py batch --config talt_evaluation/batch_configs/cnn_comparison.json
"""

import os
import sys
import argparse
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('talt_experiments')

def run_single_experiment(args):
    """Run a single experiment with the specified parameters."""
    # Start building the command
    cmd = ["python", "talt_evaluation/run_experiment.py"]
    
    # Add all arguments to the command
    for key, value in vars(args).items():
        if key == 'command':
            continue
        
        # Convert underscores to hyphens in argument names
        arg_name = key.replace('_', '-')
        
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{arg_name}")
        elif value is not None:
            cmd.append(f"--{arg_name}")
            cmd.append(str(value))
    
    # Join the command into a string
    cmd_str = " ".join(cmd)
    logger.info(f"Running command: {cmd_str}")
    
    # Ensure we're in the correct working directory
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Run the experiment
        process = subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=script_dir
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for the process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info("Experiment completed successfully!")
        else:
            logger.error(f"Experiment failed with return code {process.returncode}")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def run_batch_experiments(args):
    """Run a batch of experiments with the specified config file."""
    # Build the command
    cmd = ["python", "talt_evaluation/run_batch.py"]
    
    # Add all arguments to the command
    for key, value in vars(args).items():
        if key == 'command':
            continue
        
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif value is not None:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # Join the command into a string
    cmd_str = " ".join(cmd)
    logger.info(f"Running batch command: {cmd_str}")
    
    # Run the batch process
    process = subprocess.Popen(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    if process.returncode == 0:
        logger.info("Batch experiments completed successfully!")
    else:
        logger.error(f"Batch experiments failed with return code {process.returncode}")

def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description='Run TALT experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Create the parser for the "single" command
    single_parser = subparsers.add_parser('single', help='Run a single experiment')
    
    # Basic experiment configuration
    single_parser.add_argument('--name', type=str, default='experiment', help='Experiment name')
    single_parser.add_argument('--architecture', type=str, required=True, 
                              choices=['resnet18', 'resnet50', 'vgg16', 'efficientnet-b0', 'bert-base'],
                              help='Neural network architecture')
    single_parser.add_argument('--dataset', type=str, required=True, 
                              choices=['cifar10', 'cifar100', 'glue-sst2'],
                              help='Dataset to use for training and evaluation')
    single_parser.add_argument('--optimizer', type=str, required=True, 
                              choices=['talt', 'sgd', 'adam'],
                              help='Optimizer to use for training')
    
    # Training parameters
    single_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    single_parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    single_parser.add_argument('--lr', type=float, default=0.1, help='Base learning rate')
    single_parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    single_parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    
    # Experiment output
    single_parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    single_parser.add_argument('--save-checkpoints', action='store_true', help='Save model checkpoints')
    
    # Hardware configuration
    single_parser.add_argument('--gpu-index', type=int, default=0, help='GPU index')
    single_parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Create the parser for the "batch" command
    batch_parser = subparsers.add_parser('batch', help='Run a batch of experiments')
    
    batch_parser.add_argument('--config', type=str, required=True, 
                             help='Path to batch configuration JSON file')
    batch_parser.add_argument('--output-dir', type=str, default='./results', 
                             help='Base output directory for all experiments')
    batch_parser.add_argument('--gpu-indices', type=str, default='0', 
                             help='Comma-separated list of GPU indices to use')
    batch_parser.add_argument('--parallel', action='store_true', 
                             help='Run experiments in parallel if multiple GPUs are specified')
    batch_parser.add_argument('--max-parallel', type=int, default=None, 
                             help='Maximum number of parallel experiments')
    
    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    
    if args.command == 'single':
        run_single_experiment(args)
    elif args.command == 'batch':
        run_batch_experiments(args)

if __name__ == "__main__":
    main()