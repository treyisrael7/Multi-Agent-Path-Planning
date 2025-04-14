"""Class for visualizing algorithm comparison results"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

class ComparisonVisualizer:
    """Generates visualizations for algorithm comparisons"""
    
    def __init__(self, metrics_collector):
        """Initialize visualizer
        
        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics_collector = metrics_collector
        self.summary = metrics_collector.get_summary()
        self.performance = metrics_collector.get_performance_analysis()
        
    def plot_basic_comparison(self, save_path: str = None):
        """Plot basic performance metrics comparison
        
        Args:
            save_path: Optional path to save plot
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get data for each algorithm
        path_lengths = []
        comp_times = []
        labels = []
        
        for algorithm, stats in self.summary.items():
            if 'path_length' in stats and 'computation_time' in stats:
                path_lengths.append(stats['path_length']['mean'])
                comp_times.append(stats['computation_time']['mean'])
                labels.append(algorithm)
                
        # Plot path lengths
        ax1.bar(labels, path_lengths)
        ax1.set_title('Average Path Length')
        ax1.set_ylabel('Path Length')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot computation times
        ax2.bar(labels, comp_times)
        ax2.set_title('Average Computation Time')
        ax2.set_ylabel('Time (s)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
    def plot_performance_comparison(self, save_path: str = None):
        """Plot detailed performance comparison
        
        Args:
            save_path: Optional path to save plot
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Path length vs nodes expanded scatter
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_path_vs_nodes_scatter(ax1)
        ax1.set_title('Path Length vs Nodes Expanded')
        
        # Computation time vs nodes expanded scatter
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_time_vs_nodes_scatter(ax2)
        ax2.set_title('Computation Time vs Nodes Expanded')
        
        # CDFs
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_cdfs('path_length', ax3)
        ax3.set_title('Path Length CDF')
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_cdfs('computation_time', ax4)
        ax4.set_title('Computation Time CDF')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def _plot_path_vs_nodes_scatter(self, ax: plt.Axes):
        """Plot scatter of path length vs nodes expanded
        
        Args:
            ax: Matplotlib axes to plot on
        """
        df = self.metrics_collector.to_dataframe()
        
        for algorithm in self.metrics_collector.runs:
            algo_data = df[df['algorithm'] == algorithm]
            ax.scatter(algo_data['nodes_expanded'], 
                      algo_data['path_length'],
                      label=algorithm,
                      alpha=0.6)
            
            # Add trend line
            z = np.polyfit(algo_data['nodes_expanded'], 
                          algo_data['path_length'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(algo_data['nodes_expanded'].min(),
                                algo_data['nodes_expanded'].max(), 100)
            ax.plot(x_trend, p(x_trend), '--', alpha=0.8)
        
        ax.set_xlabel('Nodes Expanded')
        ax.set_ylabel('Path Length')
        ax.legend()
        
    def _plot_time_vs_nodes_scatter(self, ax: plt.Axes):
        """Plot scatter of computation time vs nodes expanded
        
        Args:
            ax: Matplotlib axes to plot on
        """
        df = self.metrics_collector.to_dataframe()
        
        for algorithm in self.metrics_collector.runs:
            algo_data = df[df['algorithm'] == algorithm]
            ax.scatter(algo_data['nodes_expanded'], 
                      algo_data['computation_time'],
                      label=algorithm,
                      alpha=0.6)
            
            # Add trend line
            z = np.polyfit(algo_data['nodes_expanded'], 
                          algo_data['computation_time'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(algo_data['nodes_expanded'].min(),
                                algo_data['nodes_expanded'].max(), 100)
            ax.plot(x_trend, p(x_trend), '--', alpha=0.8)
        
        ax.set_xlabel('Nodes Expanded')
        ax.set_ylabel('Computation Time (s)')
        ax.legend()
        
    def _plot_cdfs(self, metric: str, ax: plt.Axes):
        """Plot cumulative distribution function for a metric
        
        Args:
            metric: Name of metric to plot
            ax: Matplotlib axes to plot on
        """
        df = self.metrics_collector.to_dataframe()
        
        for algorithm in self.metrics_collector.runs:
            algo_data = df[df['algorithm'] == algorithm][metric]
            sorted_data = np.sort(algo_data)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cumulative, label=algorithm)
            
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
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
            
    def plot_all_comparisons(self, output_dir: str):
        """Generate summary comparison plots
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate basic comparison plot
        self.plot_basic_comparison(
            save_path=str(output_dir / 'basic_comparison.png')
        )
        
        # Generate detailed performance comparison plot
        self.plot_performance_comparison(
            save_path=str(output_dir / 'performance_comparison.png')
        )
        
        # Generate node expansion comparison plot
        self.plot_node_expansion_comparison(
            save_path=str(output_dir / 'node_expansion_comparison.png')
        ) 