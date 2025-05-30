"""Metrics collection and analysis for algorithm comparisons"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict

class MetricsCollector:
    """Collects and analyzes metrics from algorithm runs"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.runs = {}  # algorithm -> list of run metrics
        self.environments = set()  # set of environment names
        
    def add_run(self, algorithm: str, metrics: Dict[str, Any]) -> None:
        """Add metrics from a single run
        
        Args:
            algorithm: Name of algorithm
            metrics: Dictionary of metrics from run
        """
        if algorithm not in self.runs:
            self.runs[algorithm] = []
            
        # Track environment
        if 'environment' in metrics:
            self.environments.add(metrics['environment'])
            
        self.runs[algorithm].append(metrics)
        
    def get_metric_values(self, metric: str, algorithm: str, environment: str = None) -> List[float]:
        """Get all values for a specific metric
        
        Args:
            metric: Name of metric to get
            algorithm: Name of algorithm
            environment: Optional environment name to filter by
            
        Returns:
            List of values for metric
        """
        values = []
        for run in self.runs[algorithm]:
            if environment and run.get('environment') != environment:
                continue
            if metric in run:
                values.append(run[metric])
        return values
        
    def get_summary(self, metric: str, algorithm: str, environment: str = None) -> Dict[str, float]:
        """Get summary statistics for a metric
        
        Args:
            metric: Name of metric to summarize
            algorithm: Name of algorithm
            environment: Optional environment name to filter by
            
        Returns:
            Dictionary of summary statistics
        """
        values = self.get_metric_values(metric, algorithm, environment)
        if not values:
            return {}
            
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
        
    def get_efficiency_metrics(self, algorithm: str, environment: str = None) -> Dict[str, float]:
        """Calculate efficiency metrics
        
        Args:
            algorithm: Name of algorithm
            environment: Optional environment name to filter by
            
        Returns:
            Dictionary of efficiency metrics
        """
        path_lengths = self.get_metric_values('path_length', algorithm, environment)
        comp_times = self.get_metric_values('computation_time', algorithm, environment)
        nodes = self.get_metric_values('nodes_expanded', algorithm, environment)
        goals = self.get_metric_values('goals_collected', algorithm, environment)
        coverage = self.get_metric_values('coverage_percentage', algorithm, environment)
        
        if not path_lengths or not comp_times or not nodes:
            return {}
            
        metrics = {
            'path_length_per_second': np.mean(path_lengths) / np.mean(comp_times),
            'time_per_path_length': np.mean(comp_times) / np.mean(path_lengths),
            'nodes_per_path_length': np.mean(nodes) / np.mean(path_lengths)
        }
        
        # Add goal and coverage efficiency metrics if available
        if goals:
            metrics['goals_per_path_length'] = np.mean(goals) / np.mean(path_lengths)
            metrics['goals_per_second'] = np.mean(goals) / np.mean(comp_times)
            
        if coverage:
            metrics['coverage_per_path_length'] = np.mean(coverage) / np.mean(path_lengths)
            metrics['coverage_per_second'] = np.mean(coverage) / np.mean(comp_times)
            
        return metrics
        
    def save(self, filepath: str) -> None:
        """Save metrics to JSON file
        
        Args:
            filepath: Path to save file
        """
        # Convert numpy types to Python types for JSON serialization
        data = {
            'runs': self.runs,
            'environments': list(self.environments)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'MetricsCollector':
        """Load metrics from JSON file
        
        Args:
            filepath: Path to load file
            
        Returns:
            New MetricsCollector instance
        """
        collector = cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        collector.runs = data['runs']
        collector.environments = set(data.get('environments', []))
        
        return collector 