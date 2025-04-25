# Multi-Agent Pathfinding

This repository implements and compares different approaches to multi-agent pathfinding, including classical search algorithms and reinforcement learning methods.

## Repository Structure

```
.
├── algorithms/
│   ├── rl/
│   │   ├── agents/             # RL agent implementations
│   │   │   ├── dqn.py         # Deep Q-Network implementation
│   │   │   └── a2c.py         # Advantage Actor-Critic implementation
│   │   ├── environment/        # RL environment implementation
│   │   │   └── pathfinding_env.py  # Main environment class
│   │   └── training/          # Training utilities and scripts
│   └── search/                # Classical search algorithms
│       ├── astar.py          # A* implementation
│       ├── dijkstra.py       # Dijkstra's algorithm implementation
│       ├── grid_world.py     # Grid world environment for search
│       ├── path_manager.py   # Path management utilities
│       ├── movement_manager.py # Movement control utilities
│       └── configurations.py  # Environment configurations
├── comparison_results/        # Results from algorithm comparisons
│   ├── sparse/               # Results with sparse goal distribution
│   ├── dense/               # Results with dense goal distribution
│   ├── search/              # Classical search algorithm results
│   └── rl/                  # RL algorithm results
├── evaluation/
│   ├── rl/                  # RL evaluation tools
│   │   ├── scripts/         # Evaluation scripts
│   │   ├── config.json      # Configuration for evaluations
│   │   ├── compare.py       # Algorithm comparison utilities
│   │   ├── metrics.py       # Performance metrics collection
│   │   ├── visualizer.py    # Visualization tools for RL results
│   │   ├── base_evaluator.py # Base evaluation class
│   │   ├── dqn_evaluator.py # DQN-specific evaluation
│   │   └── a2c_evaluator.py # A2C-specific evaluation
│   └── search/              # Search algorithm evaluation
│       ├── scripts/         # Evaluation scripts
│       ├── metrics.py       # Performance metrics
│       ├── visualizer.py    # Visualization tools
│       ├── visualization.py # Additional visualization utilities
│       └── compare_algorithms.py # Algorithm comparison
├── models/                  # Trained model checkpoints
│   ├── dqn/                # DQN model saves
│   └── a2c/                # A2C model saves
└── utils/                  # Utility functions and tools
    ├── visualizer.py       # General visualization utilities
    ├── renderer.py         # Environment rendering tools
    └── controls.py         # User control interface
```

## Environment

The environment supports both classical search and reinforcement learning approaches:

### Grid World Environment
- Configurable grid size (default: 50x50)
- Multiple agents (default: 3)
- Configurable goals and obstacles
- Support for different goal distribution patterns
- Customizable movement patterns

### RL Environment Features
- Gymnasium-compatible interface
- 8-directional movement per agent
- 3-channel observation space (obstacles, agents, goals)
- Configurable reward structure
- Built-in metrics tracking

### Goal Distribution Patterns
- **Sparse**: Goals randomly distributed across the entire grid
- **Dense**: Goals clustered in specific areas
- Configurable via `configurations.py`

### Reward Structure
- Goal collection: +10.0
- Obstacle collision: -1.0
- Step penalty: -0.05
- Progress toward goal: +0.3
- Exploration bonus: +0.1

## Algorithms

### Reinforcement Learning
- DQN (Deep Q-Network)
  - Experience replay buffer
  - Target network for stability
  - Epsilon-greedy exploration
  - Prioritized experience replay
  - Double DQN implementation
- A2C (Advantage Actor-Critic)
  - Shared feature extraction
  - Parallel advantage estimation
  - Entropy regularization
  - N-step returns
  - GAE (Generalized Advantage Estimation)

### Classical Search
- A* with Manhattan distance heuristic
  - Optimized path finding
  - Adaptive heuristic weights
- Dijkstra's algorithm
  - Complete path coverage
  - Multi-goal path planning

## Evaluation Framework

### Metrics
- Episode return (cumulative reward)
- Steps to goals (efficiency)
- Time to convergence (learning speed)
- Total goals collected (task completion)
- Coverage area (exploration)
- Path optimality (vs. optimal solutions)
- Computational efficiency

### Visualization Tools
- Real-time environment rendering (`utils/renderer.py`)
- Performance metric plotting (`utils/visualizer.py`)
- Interactive controls (`utils/controls.py`)
- Comparative analysis plots
- Learning curve visualization
- Path visualization

### Evaluation Scripts
- Individual algorithm evaluation
- Comparative analysis
- Batch processing
- Custom metric computation
- Statistical analysis

## Usage

### Running Evaluations

All scripts should be run as Python modules from the project root directory:

```bash
# Search Algorithm Evaluations
python -m evaluation.search.run_comparison      # Compare A* and Dijkstra
python -m evaluation.search.run_search_visualization  # Visualize search process

# Reinforcement Learning Evaluations
python -m evaluation.rl.evaluate_dqn           # Evaluate DQN agent
python -m evaluation.rl.evaluate_a2c           # Evaluate A2C agent
python -m evaluation.rl.compare                # Compare RL algorithms

# Direct Evaluator Access (for custom evaluation scenarios)
python -m evaluation.rl.dqn_evaluator          # Run DQN evaluator
python -m evaluation.rl.a2c_evaluator          # Run A2C evaluator
```

Note: Do not run the Python files directly (e.g., `python path/to/script.py`), as this will cause import errors. Always use the module notation with dots instead of slashes.

### Configuration

1. Configure environment and evaluation parameters:
   - Edit `evaluation/rl/config.json` for RL settings
   - Edit `algorithms/search/configurations.py` for search settings

### Visualization Tools

The repository includes several visualization options:

```bash
# Search Algorithm Visualizations
python -m evaluation.search.run_search_visualization  # Watch search algorithms in action
python -m evaluation.search.visualize_comparisons    # View search algorithm comparisons

# RL Visualizations
python -m evaluation.rl.visualizer                   # View RL training results
python -m utils.visualizer                           # General environment visualization

# Real-time Environment Rendering
python -m utils.renderer                            # Watch agents in real-time
```

Each visualization tool offers different insights:
- `run_search_visualization`: Watch A* and Dijkstra's algorithms explore the grid
- `visualize_comparisons`: Compare path lengths, computation times, and success rates
- `visualizer`: View RL metrics like rewards, steps-to-goal, and learning curves
- `renderer`: Real-time visualization of agent behavior

### Results

Results and visualizations are saved in:
- `comparison_results/search/` - Search algorithm comparisons
- `comparison_results/rl/` - RL algorithm performance
- `comparison_results/sparse/` - Results with sparse goal distribution
- `comparison_results/dense/` - Results with dense goal distribution

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Seaborn (optional, for enhanced visualizations)

## Results

Detailed results from algorithm comparisons can be found in the `comparison_results` directory:
- Performance metrics
- Visualization outputs
- Raw data
- Statistical analyses
