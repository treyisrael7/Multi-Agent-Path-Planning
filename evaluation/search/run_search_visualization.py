"""Run and visualize pathfinding algorithms (A* and Dijkstra) in a grid world environment.

This script provides a command-line interface to run and visualize different pathfinding algorithms
with various grid world configurations. It supports both A* and Dijkstra algorithms and allows
for different environment configurations to be specified.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from algorithms.search.astar import astar_path
from algorithms.search.dijkstra import dijkstra_path
from algorithms.search.grid_world import GridWorld
from algorithms.search.configurations import CONFIGS
from utils.visualizer import Visualizer

def main():
    # Get configuration from command line arguments
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'default'
    algorithm_name = sys.argv[2] if len(sys.argv) > 2 else 'astar'
    
    # Validate and get configuration
    if config_name not in CONFIGS:
        print(f"Unknown configuration: {config_name}")
        print(f"Available configurations: {', '.join(CONFIGS.keys())}")
        return
        
    # Get pathfinding algorithm
    algorithms = {
        'astar': astar_path,
        'dijkstra': dijkstra_path
    }
    
    if algorithm_name not in algorithms:
        print(f"Unknown algorithm: {algorithm_name}")
        print(f"Available algorithms: {', '.join(algorithms.keys())}")
        return
        
    # Create environment with selected configuration and algorithm
    config = CONFIGS[config_name]()
    env = GridWorld(config=config, pathfinding_algorithm=algorithms[algorithm_name])
    
    # Print configuration details
    print(f"\nRunning simulation with:")
    print(f"- Configuration: {config_name}")
    print(f"- Grid size: {config.size}x{config.size}")
    print(f"- Agents: {config.num_agents}")
    print(f"- Goals: {config.num_goals}")
    print(f"- Obstacles: {config.num_obstacles}")
    print(f"- Algorithm: {algorithm_name}")
    print("\nControls:")
    print("- SPACE: Pause/Resume")
    print("- R: Reset")
    print("- ESC: Quit")
    
    # Create and run visualizer
    vis = Visualizer(env)
    vis.run()

if __name__ == "__main__":
    main() 