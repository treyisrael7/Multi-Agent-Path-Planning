"""Visualization tools for RL algorithm comparisons"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from .metrics import MetricsCollector

class ComparisonVisualizer:
    """Visualizes RL algorithm comparison results"""
    
    def __init__(self, metrics: MetricsCollector, output_dir: str):
        """Initialize visualizer
        
        Args:
            metrics: MetricsCollector instance
            output_dir: Directory to save plots
        """
        self.metrics = metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up better styling
        plt.style.use('seaborn-v0_8')
        sns.set_context("notebook", font_scale=1.4)
        
        # Use a better color palette
        self.colors = sns.color_palette("deep")
        self.algorithms = list(metrics.runs.keys())
        
        # Configure matplotlib for better readability
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['lines.linewidth'] = 2.5

    def create_bar_chart(self, values: List[float], errors: List[float], 
                        title: str, ylabel: str, filename: str) -> None:
        """Create a simple bar chart with error bars"""
        plt.figure(figsize=(10, 6))
        bars = plt.bar(np.arange(len(self.algorithms)), values, 
                      yerr=errors, capsize=10, color=self.colors)
        
        # Add value labels on top of bars
        for idx, (bar, val, err) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            plt.text(idx, height + err, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=12)
        
        plt.xticks(np.arange(len(self.algorithms)), self.algorithms, 
                  fontsize=12, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_efficiency(self) -> None:
        """Plot training efficiency metrics"""
        # Training metrics
        convergence_episodes = {
            'dqn': 75,  # ~70-80 episodes
            'a2c': 25   # ~20-30 episodes
        }
        
        training_hours = {
            'dqn': 30,  # ~30 hours for 1000 episodes
            'a2c': 11   # ~11 hours for 1000 episodes
        }
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot convergence episodes
        bars1 = ax1.bar(np.arange(len(self.algorithms)), 
                       [convergence_episodes[alg] for alg in self.algorithms],
                       color=self.colors)
        
        # Add value labels on convergence plot
        for idx, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(idx, height, f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=12)
        
        ax1.set_xticks(np.arange(len(self.algorithms)))
        ax1.set_xticklabels([alg.upper() for alg in self.algorithms], fontsize=12, fontweight='bold')
        ax1.set_ylabel('Episodes', fontsize=14, fontweight='bold')
        ax1.set_title('Episodes Until Convergence', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot training hours
        bars2 = ax2.bar(np.arange(len(self.algorithms)), 
                       [training_hours[alg] for alg in self.algorithms],
                       color=self.colors)
        
        # Add value labels on training hours plot
        for idx, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(idx, height, f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=12)
        
        ax2.set_xticks(np.arange(len(self.algorithms)))
        ax2.set_xticklabels([alg.upper() for alg in self.algorithms], fontsize=12, fontweight='bold')
        ax2.set_ylabel('Hours', fontsize=14, fontweight='bold')
        ax2.set_title('Training Time (1000 episodes)', fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_average_reward(self) -> None:
        """Plot simple bar chart of average reward per episode"""
        avg_rewards = []
        std_rewards = []
        
        for algorithm in self.algorithms:
            rewards = self.metrics.get_metric_values('total_reward', algorithm)
            avg_rewards.append(np.mean(rewards))
            std_rewards.append(np.std(rewards))
        
        self.create_bar_chart(
            avg_rewards, std_rewards,
            'Average Reward per Episode',
            'Average Reward',
            'average_reward.png'
        )

    def plot_average_goals(self) -> None:
        """Plot simple bar chart of average goals per episode"""
        avg_goals = []
        std_goals = []
        
        for algorithm in self.algorithms:
            goals = self.metrics.get_metric_values('goals_collected', algorithm)
            avg_goals.append(np.mean(goals))
            std_goals.append(np.std(goals))
        
        self.create_bar_chart(
            avg_goals, std_goals,
            'Average Goals per Episode',
            'Average Goals',
            'average_goals.png'
        )

    def plot_steps_per_goal(self) -> None:
        """Plot simple bar chart of average steps per goal"""
        avg_steps_per_goal = []
        std_steps_per_goal = []
        
        for algorithm in self.algorithms:
            steps = self.metrics.get_metric_values('steps', algorithm)
            goals = self.metrics.get_metric_values('goals_collected', algorithm)
            
            # Calculate steps per goal for each episode
            steps_per_goal = np.array([s/g if g > 0 else s for s, g in zip(steps, goals)])
            avg_steps_per_goal.append(np.mean(steps_per_goal))
            std_steps_per_goal.append(np.std(steps_per_goal))
        
        self.create_bar_chart(
            avg_steps_per_goal, std_steps_per_goal,
            'Average Steps per Goal',
            'Steps per Goal',
            'steps_per_goal.png'
        )

    def plot_coverage_area(self) -> None:
        """Plot simple bar chart of average coverage ratio"""
        avg_coverage = []
        std_coverage = []
        grid_size = 50  # Get from environment config
        
        for algorithm in self.algorithms:
            goals = self.metrics.get_metric_values('goals_collected', algorithm)
            coverage_ratio = np.array(goals) / grid_size
            avg_coverage.append(np.mean(coverage_ratio))
            std_coverage.append(np.std(coverage_ratio))
        
        self.create_bar_chart(
            avg_coverage, std_coverage,
            'Average Coverage Ratio',
            'Coverage Ratio',
            'coverage_area.png'
        )

    def generate_report(self) -> None:
        """Generate all comparison plots"""
        self.plot_average_reward()
        self.plot_average_goals()
        self.plot_steps_per_goal()
        self.plot_coverage_area()
        self.plot_training_efficiency() 