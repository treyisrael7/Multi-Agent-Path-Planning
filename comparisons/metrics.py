"""Metrics collection and analysis for algorithm comparisons"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any

class MetricsCollector:
    """Collects and analyzes metrics from algorithm runs"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.runs: Dict[str, List[Dict[str, float]]] = {}
        
    def add_run(self, algorithm: str, metrics: Dict[str, float]):
        """Add metrics from a single run
        
        Args:
            algorithm: Name of algorithm
            metrics: Dictionary of metric names to values
        """
        if algorithm not in self.runs:
            self.runs[algorithm] = []
        self.runs[algorithm].append(metrics)
        
    def get_metric_values(self, algorithm: str, metric: str, default: float = 0.0) -> List[float]:
        """Get all values for a metric from an algorithm
        
        Args:
            algorithm: Name of algorithm
            metric: Name of metric
            default: Default value to use if metric is missing
            
        Returns:
            List of metric values
        """
        if algorithm not in self.runs or not self.runs[algorithm]:
            return []
            
        return [run.get(metric, default) for run in self.runs[algorithm]]
        
    def get_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get summary statistics for all metrics
        
        Returns:
            Dictionary mapping algorithms to metrics to statistics
        """
        summary = {}
        
        for algorithm in self.runs:
            summary[algorithm] = {}
            
            # Get all metric names from all runs
            all_metrics = set()
            for run in self.runs[algorithm]:
                all_metrics.update(run.keys())
            
            for metric in all_metrics:
                values = self.get_metric_values(algorithm, metric)
                if values:  # Only calculate stats if we have values
                    summary[algorithm][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'percentile_25': np.percentile(values, 25),
                        'percentile_75': np.percentile(values, 75)
                    }
                
        return summary
        
    def get_time_series(self, metric: str) -> Dict[str, List[float]]:
        """Get time series data for a metric
        
        Args:
            metric: Name of metric
            
        Returns:
            Dictionary mapping algorithms to lists of values
        """
        return {
            algorithm: self.get_metric_values(algorithm, metric)
            for algorithm in self.runs
        }
        
    def get_correlation_matrix(self, algorithm: str) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for metrics of an algorithm
        
        Args:
            algorithm: Name of algorithm
            
        Returns:
            Dictionary mapping metric pairs to correlation values
        """
        if algorithm not in self.runs or not self.runs[algorithm]:
            return {}
            
        # Get all metric names from all runs
        all_metrics = set()
        for run in self.runs[algorithm]:
            all_metrics.update(run.keys())
            
        if not all_metrics:
            return {}
            
        # Create DataFrame for correlation calculation
        df = pd.DataFrame(self.runs[algorithm])
        corr_matrix = df.corr().to_dict()
        
        return corr_matrix
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame
        
        Returns:
            DataFrame with metrics
        """
        rows = []
        for algorithm in self.runs:
            for i, run in enumerate(self.runs[algorithm]):
                row = {'algorithm': algorithm, 'run': i}
                row.update(run)
                rows.append(row)
                
        return pd.DataFrame(rows)
        
    def get_performance_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed performance analysis comparing algorithms
        
        Returns:
            Dictionary with detailed performance metrics and comparisons
        """
        analysis = {}
        
        # Get basic metrics
        df = self.to_dataframe()
        
        for algorithm in self.runs:
            algo_data = df[df['algorithm'] == algorithm]
            
            # Path length analysis
            path_lengths = algo_data['path_length'].values
            path_length_stats = {
                'mean': np.mean(path_lengths),
                'std': np.std(path_lengths),
                'min': np.min(path_lengths),
                'max': np.max(path_lengths),
                'median': np.percentile(path_lengths, 50),
                'percentile_25': np.percentile(path_lengths, 25),
                'percentile_75': np.percentile(path_lengths, 75)
            }
            
            # Computation time analysis
            comp_times = algo_data['computation_time'].values
            comp_time_stats = {
                'mean': np.mean(comp_times),
                'std': np.std(comp_times),
                'min': np.min(comp_times),
                'max': np.max(comp_times),
                'median': np.percentile(comp_times, 50),
                'percentile_25': np.percentile(comp_times, 25),
                'percentile_75': np.percentile(comp_times, 75)
            }
            
            # Efficiency metrics
            efficiency = {
                'path_length_per_second': path_length_stats['mean'] / comp_time_stats['mean'],
                'time_per_path_length': comp_time_stats['mean'] / path_length_stats['mean']
            }
            
            analysis[algorithm] = {
                'path_length': path_length_stats,
                'computation_time': comp_time_stats,
                'efficiency': efficiency
            }
            
        # Add comparative analysis
        if len(self.runs) > 1:
            algorithms = list(self.runs.keys())
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    algo1, algo2 = algorithms[i], algorithms[j]
                    
                    # Path length comparison
                    path_length_diff = (
                        analysis[algo1]['path_length']['mean'] - 
                        analysis[algo2]['path_length']['mean']
                    )
                    path_length_improvement = (
                        path_length_diff / analysis[algo2]['path_length']['mean'] * 100
                    )
                    
                    # Computation time comparison
                    comp_time_diff = (
                        analysis[algo1]['computation_time']['mean'] - 
                        analysis[algo2]['computation_time']['mean']
                    )
                    comp_time_improvement = (
                        comp_time_diff / analysis[algo2]['computation_time']['mean'] * 100
                    )
                    
                    comparison_key = f'{algo1}_vs_{algo2}'
                    analysis[comparison_key] = {
                        'path_length': {
                            'difference': path_length_diff,
                            'improvement_percent': path_length_improvement
                        },
                        'computation_time': {
                            'difference': comp_time_diff,
                            'improvement_percent': comp_time_improvement
                        }
                    }
        
        return analysis
        
    def save(self, filepath: str):
        """Save metrics to a JSON file
        
        Args:
            filepath: Path to save metrics
        """
        # Convert numpy types to Python types for JSON serialization
        serializable_runs = {}
        for algorithm, runs in self.runs.items():
            serializable_runs[algorithm] = []
            for run in runs:
                serializable_run = {}
                for metric, value in run.items():
                    # Convert numpy types to Python types
                    if isinstance(value, np.integer):
                        serializable_run[metric] = int(value)
                    elif isinstance(value, np.floating):
                        serializable_run[metric] = float(value)
                    else:
                        serializable_run[metric] = value
                serializable_runs[algorithm].append(serializable_run)
                
        with open(filepath, 'w') as f:
            json.dump(serializable_runs, f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'MetricsCollector':
        """Load metrics from a JSON file
        
        Args:
            filepath: Path to load metrics from
            
        Returns:
            MetricsCollector instance
        """
        collector = cls()
        with open(filepath, 'r') as f:
            collector.runs = json.load(f)
        return collector
    
    def save_to_csv(self, filename):
        """Save metrics to a CSV file"""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        return filename
    
    def load_from_csv(self, filename):
        """Load metrics from a CSV file"""
        df = pd.read_csv(filename)
        
        # Reset metrics
        self.runs = {}
        
        # Load metrics
        for _, row in df.iterrows():
            algorithm = row['algorithm']
            run_id = row['run']
            metrics = row.drop(['algorithm', 'run']).to_dict()
            self.add_run(algorithm, metrics)
        
        return self 