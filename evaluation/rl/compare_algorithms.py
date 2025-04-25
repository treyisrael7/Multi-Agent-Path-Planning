"""Compare RL algorithms performance"""

import argparse
import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.rl.dqn_evaluator import DQNEvaluator
from evaluation.rl.a2c_evaluator import A2CEvaluator

def run_comparison(num_episodes: int = 10, visualize: bool = False) -> Dict[str, Dict[str, List[float]]]:
    """Run comparison between DQN and A2C algorithms
    
    Args:
        num_episodes: Number of episodes to evaluate
        visualize: Whether to show visualization during evaluation
        
    Returns:
        Dictionary containing metrics for each algorithm
    """
    # Initialize evaluators
    dqn_evaluator = DQNEvaluator(visualize=visualize)
    a2c_evaluator = A2CEvaluator(visualize=visualize)
    
    # Run evaluations
    print("\nEvaluating DQN...")
    dqn_metrics = dqn_evaluator.evaluate(num_episodes=num_episodes)
    
    print("\nEvaluating A2C...")
    a2c_metrics = a2c_evaluator.evaluate(num_episodes=num_episodes)
    
    return {
        'dqn': dqn_metrics,
        'a2c': a2c_metrics
    }

def plot_comparison(metrics: Dict[str, Dict[str, List[float]]], save_path: str = None):
    """Plot comparison of algorithm metrics
    
    Args:
        metrics: Dictionary containing metrics for each algorithm
        save_path: Path to save the plot (if None, show plot)
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    metrics_to_plot = ['total_reward', 'goals_collected', 'obstacles_hit']
    n_metrics = len(metrics_to_plot)
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(1, n_metrics, i)
        
        # Prepare data
        data = []
        labels = []
        for algo, algo_metrics in metrics.items():
            data.append(algo_metrics[metric])
            labels.append(algo.upper())
        
        # Plot boxplot
        sns.boxplot(data=data)
        plt.xticks(range(len(labels)), labels)
        plt.title(metric.replace('_', ' ').title())
        plt.ylabel('Value')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"\nSaved comparison plot to {save_path}")
    else:
        plt.show()

def main():
    """Main function to run comparison"""
    parser = argparse.ArgumentParser(description='Compare RL algorithms')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--no-vis', action='store_true',
                      help='Disable visualization during evaluation')
    parser.add_argument('--save-plot', type=str,
                      help='Path to save comparison plot (default: show plot)')
    
    args = parser.parse_args()
    
    # Run comparison
    metrics = run_comparison(
        num_episodes=args.episodes,
        visualize=not args.no_vis
    )
    
    # Plot results
    plot_comparison(metrics, args.save_plot)

if __name__ == '__main__':
    main() 