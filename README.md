# Pathfinding Algorithm Comparison

This project implements and compares different pathfinding algorithms (A* and Dijkstra) in various grid-based environments.

## Project Structure

```
.
├── algorithms/
│   └── pathfinding/
│       ├── astar.py         # A* implementation
│       └── dijkstra.py      # Dijkstra's implementation
├── env/
│   ├── grid_world.py        # Grid environment implementation
│   ├── configurations.py     # Environment configurations
│   ├── movement_manager.py   # Handles agent movement and collision
│   └── path_manager.py      # Manages path execution and validation
├── visualization/
│   ├── visualizer.py        # Main visualization logic
│   ├── renderer.py          # Grid rendering and drawing
│   └── controls.py          # User input handling
├── comparisons/
│   ├── metrics.py           # Metrics collection and analysis
│   ├── visualizer.py        # Comparison visualization tools
│   └── run_comparison.py    # Main comparison script
├── main.py                  # Entry point for visualization
└── requirements.txt         # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd path-planning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Pathfinding

To visualize a single pathfinding instance:

```bash
python main.py dense astar
```

Arguments:
1. Environment type: `dense` or `sparse`
2. Algorithm: `astar` or `dijkstra`

Controls:
- **SPACE**: Step through pathfinding
- **R**: Reset environment
- **ESC**: Exit

### Running Algorithm Comparisons

To run comprehensive algorithm comparisons:

```bash
python -m comparisons.run_comparison --configs dense sparse --num-runs 20
```

Options:
- `--configs`: Environment configurations to test (`dense`, `sparse`)
- `--num-runs`: Number of runs per algorithm per environment (default: 50)
- `--algorithms`: Algorithms to compare (default: both `astar` and `dijkstra`)
- `--output-dir`: Directory to save results (default: `comparison_results`)

This will:
1. Run both A* and Dijkstra's algorithm in dense and sparse environments
2. Generate comparison plots organized by environment type
3. Save results in the `comparison_results` directory

### Output Structure

```
comparison_results/
├── basic_comparison.png                    # Overall algorithm performance
├── environment_comparison_path_length.png   # Path length across environments
├── environment_comparison_computation_time.png
├── environment_comparison_nodes_expanded.png
├── dense/
│   ├── performance_comparison.png          # Detailed dense environment analysis
│   └── efficiency_comparison.png
└── sparse/
    ├── performance_comparison.png          # Detailed sparse environment analysis
    └── efficiency_comparison.png
```

### Visualization Types

1. Performance Comparisons (per environment):
   - Path Length vs Nodes Expanded
   - Computation Time vs Nodes Expanded
   - Path Length CDF
   - Computation Time CDF

2. Efficiency Metrics (per environment):
   - Path Length per Second
   - Time per Path Length
   - Nodes per Path Length

3. Environment Comparisons:
   - Path Length Distribution
   - Computation Time Distribution
   - Node Expansion Distribution

## Environment Types

- **Dense**: Grid world with high obstacle density (30% obstacles)
   - More challenging paths
   - Higher computation times
   - Good for testing algorithm efficiency

- **Sparse**: Grid world with low obstacle density (10% obstacles)
   - More direct paths available
   - Lower computation times
   - Good for baseline comparisons

## File Descriptions

### Core Components
- `main.py`: Main entry point for visualization
- `grid_world.py`: Core grid environment implementation
- `movement_manager.py`: Handles agent movement and collision detection
- `path_manager.py`: Manages path execution and validation

### Algorithms
- `astar.py`: A* pathfinding implementation with heuristic search
- `dijkstra.py`: Dijkstra's algorithm implementation

### Visualization
- `visualizer.py`: Main visualization logic and window management
- `renderer.py`: Grid rendering and graphical elements
- `controls.py`: User input handling and control mapping

### Comparison Tools
- `metrics.py`: Collects and analyzes algorithm performance metrics
- `run_comparison.py`: Runs automated comparisons between algorithms
- `visualizer.py`: Generates comparison plots and visualizations
