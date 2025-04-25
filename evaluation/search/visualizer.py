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
        
    def plot_all_comparisons(self, output_dir: str) -> None:
        """Generate all comparison plots
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic comparison - overview of performance
        self.plot_basic_comparison(str(output_dir / 'basic_comparison.png'))
        
        # Environment-specific plots
        for env in self.environments:
            env_dir = output_dir / env
            env_dir.mkdir(exist_ok=True)
            
            # Performance comparison - shows efficiency
            self.plot_performance_comparison(
                env, str(env_dir / 'performance_comparison.png'))
            
            # Goals and coverage analysis - shows effectiveness
            self.plot_goals_and_coverage(
                env, str(env_dir / 'goals_coverage_analysis.png'))
        
        print(f"Plots saved to {output_dir}")
        
    def plot_basic_comparison(self, save_path: str = None) -> None:
        """Plot basic performance metrics comparison
        
        Args:
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Algorithm Performance Overview')
        
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
        bars1 = ax1.bar(labels, path_lengths, color=self.colors)
        ax1.set_title('Average Path Length')
        ax1.set_ylabel('Path Length')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        # Plot computation times
        bars2 = ax2.bar(labels, comp_times, color=self.colors)
        ax2.set_title('Average Computation Time')
        ax2.set_ylabel('Time (s)')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on top of bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
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
            plt.close()
        else:
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
        
    def _plot_metric_boxplot(self, metric: str, environment: str, ax: plt.Axes):
        """Plot boxplot for a metric
        
        Args:
            metric: Name of metric to plot
            environment: Environment to plot metrics for
            ax: Matplotlib axes to plot on
        """
        data = []
        labels = []
        
        for algorithm in self.metrics.runs:
            values = self.metrics.get_metric_values(metric, algorithm, environment)
            if values:
                data.append(values)
                labels.append(algorithm)
                
        if not data:
            return
            
        ax.boxplot(data, labels=labels)
        ax.set_ylabel(metric.replace('_', ' ').title())
        
    def plot_node_expansion_comparison(self, environment: str, save_path: str = None):
        """Plot node expansion comparison
        
        Args:
            environment: Environment to analyze
            save_path: Optional path to save plot
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'Node Expansion Analysis - {environment}')
        
        # Nodes expanded boxplot
        self._plot_metric_boxplot('nodes_expanded', environment, ax1)
        ax1.set_title('Nodes Expanded Distribution')
        
        # Nodes visited boxplot
        self._plot_metric_boxplot('nodes_visited', environment, ax2)
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
        
    def plot_goals_and_coverage(self, environment: str, save_path: str = None) -> None:
        """Plot goals collected and coverage metrics
        
        Args:
            environment: Environment to visualize
            save_path: Optional path to save plot
        """
        # Create two separate figures for better clarity
        # Goals Analysis
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig1.suptitle(f'Goals Analysis - {environment}')
        
        # Goals vs Path Length with clearer markers and less opacity
        for algo, color in zip(self.algorithms, self.colors):
            path_lengths = self.metrics.get_metric_values('path_length', algo, environment)
            goals = self.metrics.get_metric_values('goals_collected', algo, environment)
            if path_lengths and goals:
                ax1.scatter(path_lengths, goals, label=algo, color=color, alpha=0.7, s=100)
                
                # Add trend line with confidence band
                z = np.polyfit(path_lengths, goals, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(path_lengths), max(path_lengths), 100)
                ax1.plot(x_range, p(x_range), color=color, linestyle='--')
        
        ax1.set_xlabel('Path Length')
        ax1.set_ylabel('Goals Collected')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Goals CDF with improved styling
        self._plot_cdf(ax2, 'goals_collected', 'Goals Collected', environment)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            goals_path = save_path.replace('.png', '_goals.png')
            plt.savefig(goals_path)
        plt.close()
        
        # Coverage Analysis
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle(f'Coverage Analysis - {environment}')
        
        # Coverage vs Path Length
        for algo, color in zip(self.algorithms, self.colors):
            path_lengths = self.metrics.get_metric_values('path_length', algo, environment)
            coverage = self.metrics.get_metric_values('coverage_percentage', algo, environment)
            if path_lengths and coverage:
                ax1.scatter(path_lengths, coverage, label=algo, color=color, alpha=0.7, s=100)
                
                # Add trend line with confidence band
                z = np.polyfit(path_lengths, coverage, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(path_lengths), max(path_lengths), 100)
                ax1.plot(x_range, p(x_range), color=color, linestyle='--')
        
        ax1.set_xlabel('Path Length')
        ax1.set_ylabel('Coverage (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Coverage CDF
        self._plot_cdf(ax2, 'coverage_percentage', 'Coverage (%)', environment)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            coverage_path = save_path.replace('.png', '_coverage.png')
            plt.savefig(coverage_path)
        plt.close()

    def plot_efficiency_metrics(self, environment: str, save_path: str = None) -> None:
        """Plot efficiency metrics comparison
        
        Args:
            environment: Environment to visualize
            save_path: Optional path to save plot
        """
        metrics = ['goals_per_path_length', 'goals_per_second',
                  'coverage_per_path_length', 'coverage_per_second']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Efficiency Metrics - {environment}')
        axes = axes.flatten()
        
        for ax, metric in zip(axes, metrics):
            data = []
            labels = []
            for algo in self.algorithms:
                efficiency = self.metrics.get_efficiency_metrics(algo, environment)
                if metric in efficiency:
                    data.append(efficiency[metric])
                    labels.append(algo)
            
            if data:
                ax.bar(labels, data)
                ax.set_title(metric.replace('_', ' ').title())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show() 