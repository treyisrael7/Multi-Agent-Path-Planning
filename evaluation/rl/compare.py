"""Main script for comparing RL algorithms"""

import argparse
from pathlib import Path
import json
from typing import Dict, Any, List
import numpy as np
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from evaluation.rl.metrics import MetricsCollector
from evaluation.rl.visualizer import ComparisonVisualizer
from evaluation.rl.dqn_evaluator import DQNEvaluator
from evaluation.rl.a2c_evaluator import A2CEvaluator

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        return json.load(f)
        
def run_comparison(config: Dict, output_dir: str) -> None:
    """Run comparison between specified algorithms
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get environment configuration
    env_config = config['environment']
    
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    # Run each algorithm
    for algorithm_config in config['algorithms']:
        algorithm_name = algorithm_config['name']
        print(f"\nRunning {algorithm_name}...")
        
        if algorithm_name == 'dqn':
            evaluator = DQNEvaluator(
                model_path=algorithm_config['model_path'],
                grid_size=env_config['grid_size'],
                num_agents=env_config['num_agents'],
                num_goals=env_config['num_goals'],
                num_obstacles=env_config['num_obstacles'],
                max_steps=config['max_steps'],
                visualize=config['visualize']
            )
        elif algorithm_name == 'a2c':
            evaluator = A2CEvaluator(
                model_path=algorithm_config['model_path'],
                grid_size=env_config['grid_size'],
                num_agents=env_config['num_agents'],
                num_goals=env_config['num_goals'],
                num_obstacles=env_config['num_obstacles'],
                max_steps=config['max_steps'],
                visualize=config['visualize']
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
        # Run evaluation
        results = evaluator.evaluate(num_episodes=config['num_episodes'])
        
        # Add results to metrics
        metrics.add_run(algorithm_name, {
            'environment': env_config['name'],
            **results
        })
        
        # Print summary
        print(f"\n{algorithm_name.upper()} Results:")
        print(f"Average Steps: {results['steps']:.2f}")
        print(f"Average Goals: {results['goals_collected']:.2f}")
        print(f"Average Obstacles: {results['obstacles_hit']:.2f}")
        print(f"Average Reward: {results['total_reward']:.2f}")
    
    # Save metrics
    metrics.save(os.path.join(output_dir, 'metrics.json'))
    
    # Generate visualizations
    visualizer = ComparisonVisualizer(metrics, output_dir)
    visualizer.generate_report()
    
    # Print summary
    print("\nComparison Summary:")
    for algorithm in config['algorithms']:
        print(f"\n{algorithm['name'].upper()}:")
        for metric in ['steps', 'goals_collected', 'obstacles_hit', 'total_reward']:
            summary = metrics.get_summary(metric, algorithm['name'])
            print(f"  {metric}: {summary['mean']:.2f} ± {summary['std']:.2f}")
            
        eff_metrics = metrics.get_efficiency_metrics(algorithm['name'])
        print("  Efficiency Metrics:")
        for metric, value in eff_metrics.items():
            print(f"    {metric}: {value:.4f}")
            
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compare RL algorithms")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, default="comparison_results/rl", 
                      help="Output directory (default: comparison_results/rl)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run comparison
    run_comparison(config, args.output)
    
if __name__ == "__main__":
    main() 