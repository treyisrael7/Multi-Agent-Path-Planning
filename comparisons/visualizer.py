"""Class for visualizing algorithm comparison results"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from .metrics import MetricsCollector

class ComparisonVisualizer:
    """Class for visualizing algorithm comparison results"""
    
    def __init__(self, metrics: MetricsCollector):
        """Initialize visualizer
        
        Args:
            metrics: MetricsCollector instance with results
        """
        self.metrics = metrics
        self.colors = sns.color_palette('husl', n_colors=len(metrics.runs))
        self.algorithms = list(metrics.runs.keys())
        self.environments = list(metrics.environments)
        
    def plot_basic_comparison(self, save_path: str = None) -> None:
        """Plot basic performance metrics comparison
        
        Args:
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get data for each algorithm
        path_lengths = []
        comp_times = []
        labels = []
        
        for algo in self.algorithms:
            path_length = self.metrics.get_summary('path_length', algo)
            comp_time = self.metrics.get_summary('computation_time', algo)
            
            if path_length and comp_time:
                path_lengths.append(path_length['mean'])
                comp_times.append(comp_time['mean'])
                labels.append(algo)
                
        # Plot path lengths
        ax1.bar(labels, path_lengths)
        ax1.set_title('Average Path Length')
        ax1.set_ylabel('Path Length')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot computation times
        ax2.bar(labels, comp_times)
        ax2.set_title('Average Computation Time')
        ax2.set_ylabel('Time (s)')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_performance_comparison(self, environment: str, save_path: str = None) -> None:
        """Plot performance comparison for an environment
        
        Args:
            environment: Environment to visualize
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Algorithm Performance Comparison - {environment}')
        
        # Path Length vs Nodes Expanded
        self._plot_scatter(axes[0,0], 'nodes_expanded', 'path_length',
                          'Nodes Expanded', 'Path Length', environment)
        
        # Computation Time vs Nodes Expanded
        self._plot_scatter(axes[0,1], 'nodes_expanded', 'computation_time',
                          'Nodes Expanded', 'Computation Time (s)', environment)
        
        # Path Length CDF
        self._plot_cdf(axes[1,0], 'path_length', 'Path Length', environment)
        
        # Computation Time CDF
        self._plot_cdf(axes[1,1], 'computation_time', 'Computation Time (s)', environment)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def _plot_scatter(self, ax: plt.Axes, x_metric: str, y_metric: str,
                     xlabel: str, ylabel: str, environment: str) -> None:
        """Plot scatter plot of two metrics
        
        Args:
            ax: Matplotlib axes to plot on
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            environment: Environment to plot
        """
        for algo, color in zip(self.algorithms, self.colors):
            x_values = self.metrics.get_metric_values(x_metric, algo, environment)
            y_values = self.metrics.get_metric_values(y_metric, algo, environment)
            
            if x_values and y_values:
                ax.scatter(x_values, y_values, label=algo, color=color, alpha=0.6)
                
                # Add trend line
                z = np.polyfit(x_values, y_values, 1)
                p = np.poly1d(z)
                ax.plot(x_values, p(x_values), color=color, linestyle='--', alpha=0.8)
                
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        
    def _plot_cdf(self, ax: plt.Axes, metric: str, label: str, environment: str) -> None:
        """Plot cumulative distribution function
        
        Args:
            ax: Matplotlib axes to plot on
            metric: Metric to plot
            label: Label for axis
            environment: Environment to plot
        """
        for algo, color in zip(self.algorithms, self.colors):
            values = self.metrics.get_metric_values(metric, algo, environment)
            if values:
                values = np.sort(values)
                y = np.arange(1, len(values) + 1) / len(values)
                ax.plot(values, y, label=algo, color=color)
                
        ax.set_xlabel(label)
        ax.set_ylabel('Cumulative Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_metric_boxplot(self, metric: str, ax: plt.Axes):
        """Plot boxplot for a metric
        
        Args:
            metric: Name of metric to plot
            ax: Matplotlib axes to plot on
        """
        data = []
        labels = []
        
        for algorithm in self.metrics_collector.runs:
            values = self.metrics_collector.get_metric_values(algorithm, metric)
            if values:
                data.append(values)
                labels.append(algorithm)
                
        if not data:
            return
            
        ax.boxplot(data, labels=labels)
        ax.set_ylabel(metric.replace('_', ' ').title())
        
    def plot_node_expansion_comparison(self, save_path: str = None):
        """Plot node expansion comparison
        
        Args:
            save_path: Optional path to save plot
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Nodes expanded boxplot
        self._plot_metric_boxplot('nodes_expanded', ax1)
        ax1.set_title('Nodes Expanded Distribution')
        
        # Nodes visited boxplot
        self._plot_metric_boxplot('nodes_visited', ax2)
        ax2.set_title('Nodes Visited Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_environment_comparison(self, metric: str, save_path: str = None) -> None:
        """Plot comparison of a metric across environments
        
        Args:
            metric: Metric to compare
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = []
        labels = []
        for algo in self.algorithms:
            for env in self.environments:
                values = self.metrics.get_metric_values(metric, algo, env)
                if values:
                    data.extend(values)
                    labels.extend([f'{algo}-{env}'] * len(values))
                    
        sns.boxplot(x=labels, y=data, ax=ax)
        ax.set_title(f'{metric} Comparison Across Environments')
        ax.set_xlabel('Algorithm-Environment')
        ax.set_ylabel(metric)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_efficiency_comparison(self, environment: str, save_path: str = None) -> None:
        """Plot efficiency metrics comparison
        
        Args:
            environment: Environment to analyze
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Algorithm Efficiency Comparison - {environment}')
        
        metrics = ['path_length_per_second', 'time_per_path_length', 'nodes_per_path_length']
        titles = ['Path Length per Second', 'Time per Path Length', 'Nodes per Path Length']
        
        for ax, metric, title in zip(axes, metrics, titles):
            data = []
            labels = []
            for algo in self.algorithms:
                efficiency = self.metrics.get_efficiency_metrics(algo, environment)
                if metric in efficiency:
                    data.append(efficiency[metric])
                    labels.append(algo)
                    
            sns.barplot(x=labels, y=data, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Algorithm')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_all_comparisons(self, output_dir: str) -> None:
        """Generate all comparison plots, organized by environment"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic comparison (stays in main directory)
        self.plot_basic_comparison(
            save_path=str(output_dir / 'basic_comparison.png')
        )
        
        # Environment comparisons for key metrics (stays in main directory)
        for metric in ['path_length', 'computation_time', 'nodes_expanded']:
            self.plot_environment_comparison(
                metric,
                save_path=str(output_dir / f'environment_comparison_{metric}.png')
            )
            
        # Environment-specific plots (saved in subdirectories)
        for env in self.environments:
            env_dir = output_dir / env
            env_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving plots for environment '{env}' to {env_dir}")
            
            # Performance comparison for this environment
            self.plot_performance_comparison(
                env,
                save_path=str(env_dir / f'performance_comparison.png')
            )
            
            # Efficiency comparison for this environment
            self.plot_efficiency_comparison(
                env,
                save_path=str(env_dir / f'efficiency_comparison.png')
            ) 