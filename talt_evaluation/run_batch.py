#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
import subprocess
from datetime import datetime
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('talt_batch_experiment')

def parse_args():
    parser = argparse.ArgumentParser(description='Run batch of TALT optimization experiments')
    
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to batch configuration JSON file')
    parser.add_argument('--output-dir', type=str, default='./results', 
                        help='Base output directory for all experiments')
    parser.add_argument('--gpu-indices', type=str, default='0', 
                        help='Comma-separated list of GPU indices to use')
    parser.add_argument('--parallel', action='store_true', 
                        help='Run experiments in parallel if multiple GPUs are specified')
    parser.add_argument('--max-parallel', type=int, default=None, 
                        help='Maximum number of parallel experiments')
    
    return parser.parse_args()

def run_experiment(cmd, gpu_index):
    """Run a single experiment with the specified GPU index."""
    cmd_with_gpu = cmd + f" --gpu-index {gpu_index}"
    logger.info(f"Running: {cmd_with_gpu}")
    
    # Get the current script directory to determine the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    process = subprocess.Popen(
        cmd_with_gpu,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=current_dir  # Run from the current directory to ensure proper module paths
    )
    
    stdout, stderr = process.communicate()
    success = process.returncode == 0
    
    if success:
        logger.info(f"Experiment completed successfully on GPU {gpu_index}")
    else:
        logger.error(f"Experiment failed on GPU {gpu_index}. Error: {stderr}")
    
    return {
        'cmd': cmd_with_gpu,
        'success': success,
        'gpu_index': gpu_index,
        'stdout': stdout,
        'stderr': stderr,
        'return_code': process.returncode
    }

def main():
    args = parse_args()
    
    # Load batch configuration
    with open(args.config, 'r') as f:
        batch_config = json.load(f)
    
    # Create batch output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_name = os.path.basename(args.config).replace('.json', '')
    batch_output_dir = os.path.join(args.output_dir, f"batch_{batch_name}_{timestamp}")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # Save batch configuration
    batch_config_path = os.path.join(batch_output_dir, 'batch_config.json')
    with open(batch_config_path, 'w') as f:
        json.dump(batch_config, f, indent=2)
    
    # Parse GPU indices
    gpu_indices = [int(idx) for idx in args.gpu_indices.split(',')]
    logger.info(f"Using GPUs: {gpu_indices}")
    
    # Prepare experiment commands
    experiment_cmds = []
    for exp_config in batch_config['experiments']:
        # Use full path to the run_experiment.py script
        cmd = f"python talt_evaluation/run_experiment.py"
        
        # Add experiment parameters
        for key, value in exp_config.items():
            if isinstance(value, bool) and value:
                cmd += f" --{key}"
            elif not isinstance(value, bool):
                cmd += f" --{key} {value}"
        
        # Add output directory
        cmd += f" --output-dir {batch_output_dir}"
        experiment_cmds.append(cmd)
    
    # Create batch summary file
    batch_summary = {
        'batch_name': batch_name,
        'timestamp': timestamp,
        'num_experiments': len(experiment_cmds),
        'gpu_indices': gpu_indices,
        'results': []
    }
    
    if args.parallel and len(gpu_indices) > 1:
        # Run experiments in parallel
        max_workers = min(len(gpu_indices), len(experiment_cmds))
        if args.max_parallel:
            max_workers = min(max_workers, args.max_parallel)
        
        logger.info(f"Running {len(experiment_cmds)} experiments in parallel with {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cmd = {}
            for i, cmd in enumerate(experiment_cmds):
                gpu_index = gpu_indices[i % len(gpu_indices)]
                future = executor.submit(run_experiment, cmd, gpu_index)
                future_to_cmd[future] = cmd
            
            for future in concurrent.futures.as_completed(future_to_cmd):
                result = future.result()
                batch_summary['results'].append(result)
                
                # Save updated batch summary
                batch_summary_path = os.path.join(batch_output_dir, 'batch_summary.json')
                with open(batch_summary_path, 'w') as f:
                    json.dump(batch_summary, f, indent=2)
    else:
        # Run experiments sequentially
        logger.info(f"Running {len(experiment_cmds)} experiments sequentially")
        
        for i, cmd in enumerate(experiment_cmds):
            gpu_index = gpu_indices[i % len(gpu_indices)]
            result = run_experiment(cmd, gpu_index)
            batch_summary['results'].append(result)
            
            # Save updated batch summary
            batch_summary_path = os.path.join(batch_output_dir, 'batch_summary.json')
            with open(batch_summary_path, 'w') as f:
                json.dump(batch_summary, f, indent=2)
    
    # Generate batch report
    successful_exps = sum(1 for result in batch_summary['results'] if result['success'])
    logger.info(f"Batch completed: {successful_exps}/{len(experiment_cmds)} experiments succeeded")
    logger.info(f"Batch results saved to {batch_output_dir}")

if __name__ == "__main__":
    main()