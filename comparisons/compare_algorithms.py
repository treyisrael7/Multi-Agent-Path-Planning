"""Compare different pathfinding algorithms"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from env.grid_world import GridWorld
from env.configurations import SimpleTestConfig
from algorithms.pathfinding.astar import astar_path, manhattan_distance
from algorithms.pathfinding.dijkstra import dijkstra_path
from comparisons.metrics import MetricsCollector
from comparisons.visualization import ComparisonVisualizer

def run_simulation(env, algorithm, max_steps=1000):
    """Run a single simulation with the specified algorithm"""
    metrics = {
        'path_length': 0,
        'computation_time': 0,
        'success_rate': 0,
        'steps_taken': 0
    }
    
    start_time = time.time()
    success = False
    steps = 0
    
    # Get initial positions
    agent_pos = env.agents[0].position
    goals = env.goals
    
    # Find closest goal
    closest_goal = min(goals, key=lambda g: manhattan_distance(agent_pos, g.position))
    
    # Get path using specified algorithm
    if algorithm == 'astar':
        path = astar_path(agent_pos, closest_goal.position, env.get_neighbors, env.size)
    else:  # dijkstra
        path = dijkstra_path(agent_pos, closest_goal.position, env.get_neighbors, env.size)
    
    if path:
        metrics['path_length'] = len(path)
        metrics['computation_time'] = time.time() - start_time
        metrics['success_rate'] = 1.0
        metrics['steps_taken'] = len(path)
    
    return metrics

def compare_algorithms(config, num_runs=10):
    """Compare A* and Dijkstra algorithms"""
    metrics_collector = MetricsCollector()
    visualizer = ComparisonVisualizer()
    
    algorithms = ['astar', 'dijkstra']
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        for algorithm in algorithms:
            print(f"\nTesting {algorithm.upper()}...")
            
            # Create environment
            env = GridWorld(config)
            
            # Run simulation
            metrics = run_simulation(env, algorithm)
            
            # Add metrics to collector
            metrics_collector.add_run(algorithm, metrics)
            
            print(f"Path length: {metrics['path_length']}")
            print(f"Computation time: {metrics['computation_time']:.4f} seconds")
            print(f"Success rate: {metrics['success_rate']}")
            print(f"Steps taken: {metrics['steps_taken']}")
    
    # Create visualizations and report
    report_file = visualizer.create_report(metrics_collector)
    print(f"\nComparison report generated: {report_file}")
    
    return metrics_collector

if __name__ == "__main__":
    # Create configuration
    config = SimpleTestConfig(
        size=20,
        num_agents=1,
        num_goals=1,
        obstacle_density=0.2,
        seed=42
    )
    
    # Run comparison
    metrics_collector = compare_algorithms(config, num_runs=10)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for algorithm, metrics in metrics_collector.get_summary().items():
        print(f"\n{algorithm.upper()}:")
        for metric_name, stats in metrics.items():
            print(f"{metric_name}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std:  {stats['std']:.2f}")
            print(f"  Min:  {stats['min']:.2f}")
            print(f"  Max:  {stats['max']:.2f}")
            print(f"  Median: {stats['median']:.2f}") 