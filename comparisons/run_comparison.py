"""Script to run algorithm comparisons"""

import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import time
from algorithms.pathfinding.astar import astar_path
from algorithms.pathfinding.dijkstra import dijkstra_path
from env.grid_world import GridWorld
from env.configurations import CONFIGS
from comparisons.metrics import MetricsCollector
from comparisons.visualizer import ComparisonVisualizer

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
            'nodes_visited': 0
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
    
    # Calculate metrics
    metrics = {
        'path_length': len(path) if path else float('inf'),
        'computation_time': computation_time,
        'success_rate': 1.0 if path else 0.0,
        'nodes_expanded': algo_metrics['nodes_expanded'],
        'nodes_visited': algo_metrics['nodes_visited']
    }
    
    return metrics

def run_comparison(algorithms: List[str], config_name: str, num_runs: int = 10,
                  output_dir: str = 'comparison_results') -> None:
    """Run comparison of multiple algorithms
    
    Args:
        algorithms: List of algorithm names to compare
        config_name: Name of configuration to use
        num_runs: Number of runs per algorithm
        output_dir: Directory to save results
    """
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get configuration class and create an instance
    config_class = CONFIGS[config_name]
    config = config_class()
    
    # Run each algorithm multiple times
    for algorithm in algorithms:
        print(f"\nRunning {algorithm}...")
        for run in range(num_runs):
            # Create environment
            env = GridWorld(config)
            
            # Run algorithm
            run_metrics = run_algorithm(algorithm, env, config)
            
            # Add metrics
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
    parser.add_argument('--config', type=str, default='default',
                      help='Configuration to use (default, dense, or sparse)')
    parser.add_argument('--num-runs', type=int, default=50,
                      help='Number of runs per algorithm')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    run_comparison(args.algorithms, args.config, args.num_runs, args.output_dir)

if __name__ == '__main__':
    main() 