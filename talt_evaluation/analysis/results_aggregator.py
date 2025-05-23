#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class ResultsAggregator:
    """Aggregate results from multiple experiments for analysis."""
    
    def __init__(self, results_base_dir):
        self.results_base_dir = Path(results_base_dir)
        self.all_results = []
        
    def aggregate_all_results(self):
        """Aggregate results from all experiment directories."""
        logger.info(f"Aggregating results from {self.results_base_dir}")
        
        # Find all result.json files
        result_files = list(self.results_base_dir.rglob("results.json"))
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                # Extract experiment info from path
                experiment_info = self._extract_experiment_info(result_file)
                result.update(experiment_info)
                
                # Load metadata if available
                metadata_file = result_file.parent / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    result.update(metadata)
                
                self.all_results.append(result)
                
            except Exception as e:
                logger.warning(f"Could not load results from {result_file}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)
        logger.info(f"Aggregated {len(df)} experiment results")
        
        return df
    
    def _extract_experiment_info(self, result_file):
        """Extract experiment information from file path."""
        path_parts = result_file.parts
        
        # Try to extract info from directory names
        experiment_info = {}
        
        for part in path_parts:
            if '_' in part:
                # Try to parse experiment names like "resnet18_cifar10_talt"
                components = part.split('_')
                if len(components) >= 3:
                    if any(arch in part for arch in ['resnet', 'vgg', 'efficientnet', 'bert']):
                        experiment_info['experiment_name'] = part
                        if 'resnet' in components[0]:
                            experiment_info['architecture'] = components[0]
                        elif 'efficientnet' in part:
                            experiment_info['architecture'] = f"{components[0]}-{components[1]}"
                        else:
                            experiment_info['architecture'] = components[0]
                        
                        # Extract dataset and optimizer
                        for comp in components:
                            if comp in ['cifar10', 'cifar100', 'sst2']:
                                experiment_info['dataset'] = comp
                            elif comp in ['talt', 'sgd', 'adam']:
                                experiment_info['optimizer'] = comp
        
        return experiment_info
    
    def generate_statistical_summary(self):
        """Generate statistical summary of results."""
        df = pd.DataFrame(self.all_results)
        
        if df.empty:
            return {}
        
        summary = {}
        
        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary['descriptive_stats'] = df[numeric_cols].describe().to_dict()
        
        # Group by optimizer
        if 'optimizer' in df.columns:
            optimizer_stats = df.groupby('optimizer')[numeric_cols].agg(['mean', 'std', 'count'])
            summary['optimizer_stats'] = optimizer_stats.to_dict()
            
            # Statistical significance tests between optimizers
            optimizers = df['optimizer'].unique()
            if len(optimizers) > 1 and 'test_acc' in df.columns:
                significance_tests = {}
                for i, opt1 in enumerate(optimizers):
                    for opt2 in optimizers[i+1:]:
                        acc1 = df[df['optimizer'] == opt1]['test_acc'].dropna()
                        acc2 = df[df['optimizer'] == opt2]['test_acc'].dropna()
                        
                        if len(acc1) > 1 and len(acc2) > 1:
                            t_stat, p_value = stats.ttest_ind(acc1, acc2)
                            significance_tests[f"{opt1}_vs_{opt2}"] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                
                summary['significance_tests'] = significance_tests
        
        return summary
