"""Visualization tools for algorithm comparisons"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from .metrics import MetricsCollector

class ComparisonVisualizer:
    """Generates visualizations comparing algorithm performance"""
    
    def __init__(self, metrics: MetricsCollector):
        """Initialize visualizer
        
        Args:
            metrics: MetricsCollector instance with algorithm data
        """
        self.metrics = metrics
        self.figsize = (12, 8)
        plt.style.use('seaborn')
        
    def plot_time_series(self, metric: str, title: Optional[str] = None,
                        save_path: Optional[str] = None):
        """Plot time series of a metric for all algorithms
        
        Args:
            metric: Name of metric to plot
            title: Optional plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        time_series = self.metrics.get_time_series(metric)
        for algorithm, values in time_series.items():
            plt.plot(values, label=algorithm)
            
        plt.xlabel('Run')
        plt.ylabel(metric)
        plt.title(title or f'{metric} Over Time')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_boxplots(self, metric: str, title: Optional[str] = None,
                     save_path: Optional[str] = None):
        """Plot boxplots comparing metric distributions
        
        Args:
            metric: Name of metric to plot
            title: Optional plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        data = []
        labels = []
        for algorithm in self.metrics.runs:
            values = self.metrics.get_metric_values(algorithm, metric)
            data.extend(values)
            labels.extend([algorithm] * len(values))
            
        sns.boxplot(x=labels, y=data)
        plt.xlabel('Algorithm')
        plt.ylabel(metric)
        plt.title(title or f'{metric} Distribution by Algorithm')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_correlation_heatmap(self, algorithm: str, title: Optional[str] = None,
                               save_path: Optional[str] = None):
        """Plot correlation heatmap for metrics of an algorithm
        
        Args:
            algorithm: Name of algorithm
            title: Optional plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        corr_matrix = self.metrics.get_correlation_matrix(algorithm)
        metrics = list(corr_matrix.keys())
        
        # Convert to numpy array for heatmap
        corr_array = np.zeros((len(metrics), len(metrics)))
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                corr_array[i, j] = corr_matrix[metric1][metric2]
                
        sns.heatmap(corr_array, annot=True, cmap='coolwarm', 
                   xticklabels=metrics, yticklabels=metrics)
        plt.title(title or f'Metric Correlations - {algorithm}')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_summary_stats(self, metric: str, title: Optional[str] = None,
                         save_path: Optional[str] = None):
        """Plot summary statistics for a metric
        
        Args:
            metric: Name of metric to plot
            title: Optional plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        summary = self.metrics.get_summary()
        algorithms = list(summary.keys())
        
        means = [summary[algo][metric]['mean'] for algo in algorithms]
        stds = [summary[algo][metric]['std'] for algo in algorithms]
        
        x = np.arange(len(algorithms))
        plt.bar(x, means, yerr=stds, capsize=5)
        plt.xticks(x, algorithms)
        plt.ylabel(metric)
        plt.title(title or f'{metric} Summary Statistics')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_report(self, output_dir: str):
        """Generate complete comparison report
        
        Args:
            output_dir: Directory to save plots
        """
        # Get all metrics from first algorithm
        first_algo = list(self.metrics.runs.keys())[0]
        metrics = list(self.metrics.runs[first_algo][0].keys())
        
        # Generate plots for each metric
        for metric in metrics:
            # Time series
            self.plot_time_series(
                metric,
                save_path=f'{output_dir}/{metric}_time_series.png'
            )
            
            # Boxplots
            self.plot_boxplots(
                metric,
                save_path=f'{output_dir}/{metric}_boxplots.png'
            )
            
            # Summary stats
            self.plot_summary_stats(
                metric,
                save_path=f'{output_dir}/{metric}_summary.png'
            )
            
        # Generate correlation heatmaps for each algorithm
        for algorithm in self.metrics.runs:
            self.plot_correlation_heatmap(
                algorithm,
                save_path=f'{output_dir}/{algorithm}_correlations.png'
            )

    def plot_metric_comparison(self, metrics_collector, metric_name, title=None, figsize=(10, 6)):
        """Plot comparison of a specific metric across algorithms"""
        summary = metrics_collector.get_summary()
        
        # Extract data for the specified metric
        algorithms = []
        means = []
        stds = []
        
        for algorithm, metrics in summary.items():
            if metric_name in metrics:
                algorithms.append(algorithm)
                means.append(metrics[metric_name]['mean'])
                stds.append(metrics[metric_name]['std'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bars
        x = np.arange(len(algorithms))
        bars = ax.bar(x, means, yerr=stds, capsize=10)
        
        # Add labels
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(title or f'Comparison of {metric_name.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels([alg.upper() for alg in algorithms])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        filename = self.output_dir / f"{metric_name}_comparison.png"
        plt.savefig(filename)
        plt.close()
        
        return filename
    
    def plot_all_metrics(self, metrics_collector, figsize=(12, 8)):
        """Plot all metrics in a grid"""
        summary = metrics_collector.get_summary()
        
        # Get all unique metrics
        all_metrics = set()
        for algorithm_metrics in summary.values():
            all_metrics.update(algorithm_metrics.keys())
        
        # Determine grid size
        n_metrics = len(all_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric_name in enumerate(sorted(all_metrics)):
            ax = axes[i]
            
            # Extract data
            algorithms = []
            means = []
            stds = []
            
            for algorithm, metrics in summary.items():
                if metric_name in metrics:
                    algorithms.append(algorithm)
                    means.append(metrics[metric_name]['mean'])
                    stds.append(metrics[metric_name]['std'])
            
            # Plot bars
            x = np.arange(len(algorithms))
            bars = ax.bar(x, means, yerr=stds, capsize=5)
            
            # Add labels
            ax.set_xlabel('Algorithm')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels([alg.upper() for alg in algorithms], rotation=45)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        for i in range(len(all_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        filename = self.output_dir / "all_metrics_comparison.png"
        plt.savefig(filename)
        plt.close()
        
        return filename
    
    def create_report(self, metrics_collector, title="Algorithm Comparison Report"):
        """Create a comprehensive HTML report"""
        summary = metrics_collector.get_summary()
        
        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # Add summary tables
        for algorithm, metrics in summary.items():
            html += f"<h2>{algorithm.upper()} Algorithm</h2>"
            html += "<table>"
            html += "<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Median</th></tr>"
            
            for metric_name, stats in metrics.items():
                html += f"<tr>"
                html += f"<td>{metric_name.replace('_', ' ').title()}</td>"
                html += f"<td>{stats['mean']:.2f}</td>"
                html += f"<td>{stats['std']:.2f}</td>"
                html += f"<td>{stats['min']:.2f}</td>"
                html += f"<td>{stats['max']:.2f}</td>"
                html += f"<td>{stats['median']:.2f}</td>"
                html += f"</tr>"
            
            html += "</table>"
        
        # Add visualizations
        html += "<h2>Visualizations</h2>"
        
        # Plot all metrics
        all_metrics_plot = self.plot_all_metrics(metrics_collector)
        if all_metrics_plot:
            html += f"<div class='metric'>"
            html += f"<h3>All Metrics Comparison</h3>"
            html += f"<img src='{all_metrics_plot.name}' alt='All Metrics Comparison'>"
            html += f"</div>"
        
        # Plot individual metrics
        for algorithm, metrics in summary.items():
            for metric_name in metrics.keys():
                # Time series
                time_series_plot = self.plot_time_series(metric_name)
                if time_series_plot:
                    html += f"<div class='metric'>"
                    html += f"<h3>{metric_name.replace('_', ' ').title()} Time Series</h3>"
                    html += f"<img src='{time_series_plot.name}' alt='{metric_name} Time Series'>"
                    html += f"</div>"
                
                # Comparison
                comparison_plot = self.plot_metric_comparison(metrics_collector, metric_name)
                if comparison_plot:
                    html += f"<div class='metric'>"
                    html += f"<h3>{metric_name.replace('_', ' ').title()} Comparison</h3>"
                    html += f"<img src='{comparison_plot.name}' alt='{metric_name} Comparison'>"
                    html += f"</div>"
        
        html += """
        </body>
        </html>
        """
        
        # Save HTML report
        report_file = self.output_dir / "comparison_report.html"
        with open(report_file, 'w') as f:
            f.write(html)
        
        return report_file 