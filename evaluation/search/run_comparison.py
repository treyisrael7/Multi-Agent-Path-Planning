"""Script to run algorithm comparisons"""

import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import time
import numpy as np
from algorithms.pathfinding.astar import astar_path
from algorithms.pathfinding.dijkstra import dijkstra_path
from env.grid_world import GridWorld
from env.configurations import CONFIGS
from .metrics import MetricsCollector
from .visualizer import ComparisonVisualizer

def run_algorithm(algorithm: str, env: GridWorld, config: Any) -> Dict[str, Any]:
    """Run a single algorithm and collect metrics
    
    Args:
        algorithm: Name of algorithm to run
        env: GridWorld environment
        config: Configuration object
        
    Returns:
        Dictionary of metrics
    """
    # Get the first agent and goal for testing
    if not env.agent_positions or not env.goal_positions:
        return {
            'path_length': float('inf'),
            'computation_time': 0.0,
            'success_rate': 0.0,
            'nodes_expanded': 0,
            'nodes_visited': 0,
            'goals_collected': 0,
            'coverage_percentage': 0.0
        }
    
    start = env.agent_positions[0]
    goal = env.goal_positions[0]
    
    start_time = time.time()
    
    if algorithm == 'astar':
        path, algo_metrics = astar_path(start, goal, env.get_neighbors, env.size)
    elif algorithm == 'dijkstra':
        path, algo_metrics = dijkstra_path(start, goal, env.get_neighbors, env.size)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    computation_time = time.time() - start_time
    
    # Track visited cells and goals collected
    visited_cells = set([start])
    goals_collected = 0
    if path:
        for pos in path:
            visited_cells.add(pos)
            if pos in env.goal_positions:
                goals_collected += 1
    
    # Calculate coverage (excluding obstacles)
    total_traversable_cells = env.size * env.size - np.sum(env.grid == 1)  # Count of non-obstacle cells
    coverage_percentage = (len(visited_cells) / total_traversable_cells) * 100
    
    # Calculate metrics
    metrics = {
        'path_length': len(path) if path else float('inf'),
        'computation_time': computation_time,
        'success_rate': 1.0 if path else 0.0,
        'nodes_expanded': algo_metrics['nodes_expanded'],
        'nodes_visited': algo_metrics['nodes_visited'],
        'goals_collected': goals_collected,
        'coverage_percentage': coverage_percentage
    }
    
    return metrics

def run_comparison(algorithms: List[str], config_names: List[str], num_runs: int = 10,
                  output_dir: str = 'comparison_results') -> None:
    """Run comparison of multiple algorithms across different environments
    
    Args:
        algorithms: List of algorithm names to compare
        config_names: List of configuration names to test
        num_runs: Number of runs per algorithm per environment
        output_dir: Directory to save results
    """
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run each algorithm in each environment multiple times
    for config_name in config_names:
        print(f"\nTesting environment: {config_name}")
        
        # Get configuration class and create an instance
        config_class = CONFIGS[config_name]
        config = config_class()
        
        for algorithm in algorithms:
            print(f"\nRunning {algorithm}...")
            for run in range(num_runs):
                # Create environment
                env = GridWorld(config)
                
                # Run algorithm
                run_metrics = run_algorithm(algorithm, env, config)
                
                # Add metrics with environment info
                run_metrics['environment'] = config_name
                metrics.add_run(algorithm, run_metrics)
                
                print(f"Run {run + 1}/{num_runs}: "
                      f"Path length = {run_metrics['path_length']}, "
                      f"Time = {run_metrics['computation_time']:.3f}s, "
                      f"Success = {run_metrics['success_rate']}")
                
    # Generate visualizations
    visualizer = ComparisonVisualizer(metrics)
    visualizer.plot_all_comparisons(output_dir)
    
    # Save metrics
    metrics.save(str(output_dir / 'metrics.json'))
    
    print(f"\nResults saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run algorithm comparisons')
    parser.add_argument('--algorithms', type=str, nargs='+',
                      default=['astar', 'dijkstra'],
                      help='Algorithms to compare')
    parser.add_argument('--configs', type=str, nargs='+',
                      default=['dense', 'sparse'],
                      help='Configurations to test')
    parser.add_argument('--num-runs', type=int, default=50,
                      help='Number of runs per algorithm per environment')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    run_comparison(args.algorithms, args.configs, args.num_runs, args.output_dir)

if __name__ == '__main__':
    main() 