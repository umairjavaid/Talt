#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive TALT Evaluation Orchestrator

This script orchestrates comprehensive TALT comparisons by:
1. Using existing batch experiment system from run_experiments.py
2. Generating additional experiment configurations
3. Extending existing visualization framework
4. Providing cross-experiment analysis and reporting
"""

import os
import sys
import json
import subprocess
import logging
import argparse
import glob
from datetime import datetime
from pathlib import Path
import shutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('talt_comprehensive_evaluation')

class ComprehensiveTALTEvaluator:
    """Orchestrates comprehensive TALT evaluation using existing infrastructure."""
    
    def __init__(self, output_dir="./comprehensive_results", gpu_indices="0", parallel=False, 
                 max_parallel=None, configs_to_run=None):
        self.output_dir = Path(output_dir)
        self.gpu_indices = gpu_indices
        self.parallel = parallel
        self.max_parallel = max_parallel
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = self.output_dir / f"comprehensive_evaluation_{self.timestamp}"
        self.configs_to_run = configs_to_run or ["all"]
        
        # Ensure output directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get paths to config directories and analysis modules
        self.config_dir = Path("talt_evaluation/batch_configs")
        self.analysis_dir = Path("talt_evaluation/analysis")
        
        # Track experiment results
        self.experiments_run = []
        self.experiment_results = {}
        
        # Create experiment directories
        self.config_output_dirs = {}
        
        logger.info(f"Initialized comprehensive TALT evaluation in {self.results_dir}")
        logger.info(f"Using GPU indices: {self.gpu_indices}")
        logger.info(f"Parallel execution: {self.parallel}")
        
    def _load_existing_configs(self):
        """Load existing batch configuration files."""
        configs = {}
        
        # Find all JSON config files in the batch_configs directory
        config_files = list(self.config_dir.glob("*.json"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                configs[config_file.stem] = {
                    'path': config_file,
                    'data': config_data
                }
                logger.info(f"Loaded existing config: {config_file.name}")
            except Exception as e:
                logger.error(f"Error loading config {config_file}: {e}")
        
        return configs
    
    def _generate_comprehensive_configs(self):
        """Generate additional configurations for comprehensive evaluation."""
        # Load existing configs to extend
        existing_configs = self._load_existing_configs()
        generated_configs = {}
        
        # Generate directory for new configs
        config_gen_dir = self.results_dir / "generated_configs"
        config_gen_dir.mkdir(exist_ok=True)
        
        # Process CNN comparison configs
        if "cnn_comparison" in existing_configs:
            # Create sensitivity analysis config
            cnn_sensitivity = self._generate_sensitivity_config(
                existing_configs["cnn_comparison"]["data"], 
                "cnn_sensitivity"
            )
            sensitivity_path = config_gen_dir / "cnn_sensitivity.json"
            with open(sensitivity_path, 'w') as f:
                json.dump(cnn_sensitivity, f, indent=2)
            
            generated_configs["cnn_sensitivity"] = {
                'path': sensitivity_path,
                'data': cnn_sensitivity
            }
            
            # Create ablation study config
            cnn_ablation = self._generate_ablation_config(
                existing_configs["cnn_comparison"]["data"], 
                "cnn_ablation"
            )
            ablation_path = config_gen_dir / "cnn_ablation.json"
            with open(ablation_path, 'w') as f:
                json.dump(cnn_ablation, f, indent=2)
            
            generated_configs["cnn_ablation"] = {
                'path': ablation_path,
                'data': cnn_ablation
            }
            
            # Create scaling config
            cnn_scaling = self._generate_scaling_config(
                existing_configs["cnn_comparison"]["data"], 
                "cnn_scaling"
            )
            scaling_path = config_gen_dir / "cnn_scaling.json"
            with open(scaling_path, 'w') as f:
                json.dump(cnn_scaling, f, indent=2)
            
            generated_configs["cnn_scaling"] = {
                'path': scaling_path,
                'data': cnn_scaling
            }
        
        # Process BERT comparison configs if they exist
        if "bert_comparison" in existing_configs:
            # Create BERT sensitivity config
            bert_sensitivity = self._generate_sensitivity_config(
                existing_configs["bert_comparison"]["data"], 
                "bert_sensitivity",
                is_bert=True
            )
            bert_sensitivity_path = config_gen_dir / "bert_sensitivity.json"
            with open(bert_sensitivity_path, 'w') as f:
                json.dump(bert_sensitivity, f, indent=2)
            
            generated_configs["bert_sensitivity"] = {
                'path': bert_sensitivity_path,
                'data': bert_sensitivity
            }
        
        # Merge existing and generated configs
        all_configs = {**existing_configs, **generated_configs}
        
        # Log generated configs
        logger.info(f"Generated {len(generated_configs)} additional config files in {config_gen_dir}")
        
        return all_configs
    
    def _generate_sensitivity_config(self, base_config, name, is_bert=False):
        """Generate hyperparameter sensitivity configurations based on a base config."""
        sensitivity_config = {
            "description": f"Hyperparameter sensitivity analysis for {name}",
            "base_config": base_config.get("description", ""),
            "experiments": []
        }
        
        # Find TALT experiments in base config to extend
        talt_experiments = [exp for exp in base_config["experiments"] 
                          if exp.get("optimizer", "") == "talt"]
        
        if not talt_experiments:
            logger.warning(f"No TALT experiments found in base config for {name}")
            return sensitivity_config
        
        # Use the first TALT experiment as a base
        base_talt_exp = talt_experiments[0]
        
        # Generate sensitivity experiments for learning rate
        lr_values = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1] if not is_bert else [1e-5, 2e-5, 5e-5, 1e-4]
        for lr in lr_values:
            lr_exp = base_talt_exp.copy()
            lr_exp["name"] = f"{base_talt_exp['name']}_lr_{lr}"
            lr_exp["lr"] = lr
            sensitivity_config["experiments"].append(lr_exp)
        
        # Generate sensitivity experiments for valley_strength
        valley_strengths = [0.05, 0.1, 0.2, 0.3, 0.5]
        for vs in valley_strengths:
            vs_exp = base_talt_exp.copy()
            vs_exp["name"] = f"{base_talt_exp['name']}_vs_{vs}"
            vs_exp["valley_strength"] = vs
            sensitivity_config["experiments"].append(vs_exp)
        
        # Generate sensitivity experiments for smoothing_factor
        smoothing_factors = [0.1, 0.3, 0.5, 0.7]
        for sf in smoothing_factors:
            sf_exp = base_talt_exp.copy()
            sf_exp["name"] = f"{base_talt_exp['name']}_sf_{sf}"
            sf_exp["smoothing_factor"] = sf
            sensitivity_config["experiments"].append(sf_exp)
            
        return sensitivity_config
    
    def _generate_ablation_config(self, base_config, name):
        """Generate ablation study configurations."""
        ablation_config = {
            "description": f"Ablation studies for {name}",
            "base_config": base_config.get("description", ""),
            "experiments": []
        }
        
        # Find TALT experiments in base config to extend
        talt_experiments = [exp for exp in base_config["experiments"] 
                          if exp.get("optimizer", "") == "talt"]
        
        if not talt_experiments:
            logger.warning(f"No TALT experiments found in base config for {name}")
            return ablation_config
        
        # Use the first TALT experiment as a base
        base_talt_exp = talt_experiments[0]
        
        # Generate ablation for different TALT components
        
        # 1. No valley detection
        no_valley = base_talt_exp.copy()
        no_valley["name"] = f"{base_talt_exp['name']}_no_valley"
        no_valley["valley_strength"] = 0.0
        ablation_config["experiments"].append(no_valley)
        
        # 2. No curvature smoothing
        no_smoothing = base_talt_exp.copy()
        no_smoothing["name"] = f"{base_talt_exp['name']}_no_smoothing"
        no_smoothing["smoothing_factor"] = 0.0
        ablation_config["experiments"].append(no_smoothing)
        
        # 3. Small projection dimension
        small_proj = base_talt_exp.copy()
        small_proj["name"] = f"{base_talt_exp['name']}_small_proj"
        small_proj["projection_dim"] = 8
        ablation_config["experiments"].append(small_proj)
        
        # 4. Large projection dimension
        large_proj = base_talt_exp.copy()
        large_proj["name"] = f"{base_talt_exp['name']}_large_proj"
        large_proj["projection_dim"] = 128
        ablation_config["experiments"].append(large_proj)
        
        # 5. No adaptive regularization
        no_adaptive_reg = base_talt_exp.copy()
        no_adaptive_reg["name"] = f"{base_talt_exp['name']}_no_adaptive_reg"
        no_adaptive_reg["adaptive_reg"] = False
        ablation_config["experiments"].append(no_adaptive_reg)
        
        return ablation_config
    
    def _generate_scaling_config(self, base_config, name):
        """Generate scaling configurations to test across different model sizes and datasets."""
        scaling_config = {
            "description": f"Scaling experiments for {name}",
            "base_config": base_config.get("description", ""),
            "experiments": []
        }
        
        # Find experiments with different optimizers
        optimizers = set()
        for exp in base_config["experiments"]:
            if "optimizer" in exp:
                optimizers.add(exp["optimizer"])
        
        # Model architectures to test
        architectures = ["resnet18", "resnet50", "vgg16", "efficientnet-b0"]
        
        # Datasets to test
        datasets = ["cifar10", "cifar100"]
        
        # Create scaled experiments for each optimizer
        for opt in optimizers:
            for arch in architectures:
                for dataset in datasets:
                    # Skip some combinations to reduce total experiments
                    if arch == "efficientnet-b0" and dataset == "cifar100":
                        continue
                    
                    new_exp = {
                        "name": f"{arch}_{dataset}_{opt}",
                        "architecture": arch,
                        "dataset": dataset,
                        "optimizer": opt,
                        "epochs": 10,  # Reduce epochs for scaling experiments
                        "batch-size": 128,
                        "mixed-precision": True
                    }
                    
                    # Add optimizer-specific settings
                    if opt == "talt":
                        new_exp["lr"] = 0.01
                        new_exp["valley_strength"] = 0.2
                        new_exp["smoothing_factor"] = 0.3
                    elif opt == "sgd":
                        new_exp["lr"] = 0.1
                        new_exp["momentum"] = 0.9
                    elif opt == "adam":
                        new_exp["lr"] = 0.001
                    
                    scaling_config["experiments"].append(new_exp)
        
        return scaling_config
    
    def run_all_experiments(self):
        """Run all experiment configurations using existing batch system."""
        # Generate comprehensive configs
        all_configs = self._generate_comprehensive_configs()
        
        # Filter configs based on user selection
        configs_to_process = {}
        if "all" in self.configs_to_run:
            configs_to_process = all_configs
        else:
            for config_name in self.configs_to_run:
                if config_name in all_configs:
                    configs_to_process[config_name] = all_configs[config_name]
                else:
                    logger.warning(f"Requested config '{config_name}' not found. Skipping.")
        
        if not configs_to_process:
            logger.error("No valid configurations to run!")
            return
        
        logger.info(f"Running {len(configs_to_process)} configurations: {', '.join(configs_to_process.keys())}")
        
        # Run each configuration
        for config_name, config_info in configs_to_process.items():
            config_path = config_info['path']
            output_dir = self.results_dir / config_name
            self.config_output_dirs[config_name] = output_dir
            output_dir.mkdir(exist_ok=True)
            
            logger.info(f"Running batch: {config_name} ({config_path})")
            
            # Prepare command to run batch using existing run_experiments.py
            cmd = [
                sys.executable,
                "run_experiments.py",
                "batch",
                "--config", str(config_path),
                "--output-dir", str(output_dir),
                "--gpu-indices", self.gpu_indices,
            ]
            
            # Add parallel flag if specified
            if self.parallel:
                cmd.append("--parallel")
                
            # Add max_parallel if specified
            if self.max_parallel:
                cmd.extend(["--max-parallel", str(self.max_parallel)])
            
            # Log the command
            cmd_str = " ".join(cmd)
            logger.info(f"Executing: {cmd_str}")
            
            # Save the command to a log file
            with open(output_dir / "command.log", "w") as f:
                f.write(f"{cmd_str}\n")
            
            # Execute the command
            try:
                start_time = time.time()
                process = subprocess.Popen(
                    cmd_str,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # Record that this configuration was run
                self.experiments_run.append({
                    'config_name': config_name,
                    'config_path': str(config_path),
                    'output_dir': str(output_dir),
                    'start_time': start_time
                })
                
                # Stream output to console and log file
                with open(output_dir / "output.log", "w") as log_file:
                    for line in process.stdout:
                        sys.stdout.write(line)
                        log_file.write(line)
                
                process.wait()
                end_time = time.time()
                
                if process.returncode == 0:
                    logger.info(f"Successfully completed batch {config_name} in {end_time - start_time:.2f} seconds")
                    # Update the experiment record with completion info
                    self.experiments_run[-1].update({
                        'success': True,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
                else:
                    logger.error(f"Batch {config_name} failed with return code {process.returncode}")
                    self.experiments_run[-1].update({
                        'success': False,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'error_code': process.returncode
                    })
            
            except Exception as e:
                logger.error(f"Error executing batch {config_name}: {e}")
                self.experiments_run[-1].update({
                    'success': False,
                    'error': str(e)
                })
        
        # Save summary of all experiments run
        with open(self.results_dir / "comprehensive_experiments.json", "w") as f:
            json.dump({
                'timestamp': self.timestamp,
                'configurations': self.experiments_run
            }, f, indent=2)
        
        logger.info(f"Completed {len(self.experiments_run)} batches")
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis using existing visualization framework."""
        logger.info("Generating comprehensive analysis...")
        
        # Import the results aggregator and adaptive visualizer
        try:
            # Add talt_evaluation to path
            talt_eval_path = str(Path.cwd() / "talt_evaluation")
            if talt_eval_path not in sys.path:
                sys.path.insert(0, talt_eval_path)
                
            from analysis.results_aggregator import ResultsAggregator
            from visualization.adaptive_visualizer import AdaptiveVisualizer
        except ImportError as e:
            logger.error(f"Failed to import analysis modules: {e}")
            return
        
        # Create analysis directory
        analysis_dir = self.results_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Create comprehensive adaptive visualizer
        comprehensive_visualizer = AdaptiveVisualizer(
            output_dir=analysis_dir,
            experiment_name="comprehensive_evaluation"
        )
        
        # Aggregate results from all experiments
        try:
            aggregator = ResultsAggregator(self.results_dir)
            all_results_df = aggregator.aggregate_all_results()
            
            # Save the aggregated results
            all_results_df.to_csv(analysis_dir / "all_results.csv", index=False)
            
            # Generate statistical summary
            stats_summary = aggregator.generate_statistical_summary()
            with open(analysis_dir / "statistical_summary.json", "w") as f:
                json.dump(stats_summary, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            # Create minimal data for visualization
            all_results_df = None
            stats_summary = {}
        
        # Prepare comprehensive experiment data
        comprehensive_data = {
            'has_batch_data': True,
            'has_comparison_data': True,
            'configurations': self.experiments_run,
            'optimizer_types': ['talt', 'sgd', 'adam'],  # Common optimizers
            'architecture_types': ['cnn', 'transformer']
        }
        
        if all_results_df is not None:
            comprehensive_data['aggregated_results'] = all_results_df.to_dict('records')
            comprehensive_data['statistical_summary'] = stats_summary
        
        # Generate comprehensive visualizations
        try:
            comprehensive_viz = comprehensive_visualizer.generate_all_visualizations(comprehensive_data)
            
            viz_summary = comprehensive_visualizer.get_visualization_summary()
            logger.info(f"Generated comprehensive analysis with {viz_summary['total_visualizations']} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating comprehensive visualizations: {e}")
            logger.debug(traceback.format_exc())
        
        logger.info(f"Comprehensive analysis completed and saved to {analysis_dir}")
        
        return analysis_dir
    
    def _generate_report(self, analysis_dir, results_df, stats_summary):
        """Generate a comprehensive HTML report."""
        try:
            import pandas as pd
            from jinja2 import Template
            
            # Create report template
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>TALT Comprehensive Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1, h2, h3 { color: #333366; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .chart-container { margin: 20px 0; }
                    .best-result { font-weight: bold; color: green; }
                    .footer { margin-top: 50px; font-size: small; color: #666; }
                </style>
            </head>
            <body>
                <h1>TALT Comprehensive Evaluation Report</h1>
                <p>Generated on: {{ timestamp }}</p>
                
                <h2>Overview</h2>
                <p>This report summarizes the results of {{ num_experiments }} experiments across {{ num_configs }} configurations.</p>
                
                <h2>Top Performing Experiments</h2>
                {{ top_experiments_html }}
                
                <h2>Optimizer Comparison</h2>
                <div class="chart-container">
                    <img src="optimizer_performance.png" alt="Optimizer Performance Matrix" style="max-width: 100%;">
                </div>
                
                <h2>Efficiency Analysis</h2>
                <div class="chart-container">
                    <img src="efficiency_analysis.png" alt="Efficiency Analysis" style="max-width: 100%;">
                </div>
                
                <h2>Statistical Summary</h2>
                <h3>Average Performance by Optimizer</h3>
                {{ avg_by_optimizer_html }}
                
                <h3>Statistical Tests</h3>
                <pre>{{ statistical_tests }}</pre>
                
                <h2>Experiment Details</h2>
                {{ all_results_html }}
                
                <div class="footer">
                    <p>Generated by TALT Comprehensive Evaluation Framework</p>
                </div>
            </body>
            </html>
            """
            
            # Get top experiments
            top_experiments = results_df.sort_values(by='test_acc', ascending=False).head(10)
            top_experiments_html = top_experiments.to_html(classes='dataframe')
            
            # Get average performance by optimizer
            avg_by_optimizer = results_df.groupby('optimizer').mean().reset_index()
            avg_by_optimizer_html = avg_by_optimizer.to_html(classes='dataframe')
            
            # Render template
            template = Template(template_str)
            html = template.render(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                num_experiments=len(results_df),
                num_configs=len(self.config_output_dirs),
                top_experiments_html=top_experiments_html,
                avg_by_optimizer_html=avg_by_optimizer_html,
                statistical_tests=json.dumps(stats_summary.get('significance_tests', {}), indent=2),
                all_results_html=results_df.to_html(classes='dataframe')
            )
            
            # Write HTML report
            with open(analysis_dir / "comprehensive_report.html", "w") as f:
                f.write(html)
            
            logger.info(f"Generated HTML report: {analysis_dir / 'comprehensive_report.html'}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run comprehensive TALT evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--output-dir', type=str, default='./comprehensive_results',
                        help='Base directory for results')
    parser.add_argument('--gpu-indices', type=str, default='0',
                        help='Comma-separated list of GPU indices to use')
    parser.add_argument('--parallel', action='store_true',
                        help='Run experiments in parallel if multiple GPUs are specified')
    parser.add_argument('--max-parallel', type=int, default=None,
                        help='Maximum number of parallel experiments')
    parser.add_argument('--configs', type=str, default='all',
                        help='Comma-separated list of config names to run, or "all"')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Skip running experiments and only generate analysis')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='For --analysis-only, specify the directory containing results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse configs to run
    configs_to_run = args.configs.split(',') if args.configs != 'all' else ['all']
    
    if args.analysis_only:
        if args.results_dir is None:
            logger.error("--results-dir must be specified when using --analysis-only")
            return
        
        # Run analysis only on existing results
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            logger.error(f"Results directory does not exist: {results_dir}")
            return
        
        # Create evaluator with the existing results directory
        evaluator = ComprehensiveTALTEvaluator(output_dir=str(results_dir.parent))
        evaluator.results_dir = results_dir
        
        # Generate analysis
        evaluator.generate_comprehensive_analysis()
        
    else:
        # Create evaluator and run experiments
        evaluator = ComprehensiveTALTEvaluator(
            output_dir=args.output_dir,
            gpu_indices=args.gpu_indices,
            parallel=args.parallel,
            max_parallel=args.max_parallel,
            configs_to_run=configs_to_run
        )
        
        # Run all experiments
        evaluator.run_all_experiments()
        
        # Generate analysis
        evaluator.generate_comprehensive_analysis()
    
    logger.info("Comprehensive evaluation completed")

if __name__ == "__main__":
    main()
