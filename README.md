# Multi-Agent Pathfinding Simulation

A Python-based visualization tool for multi-agent pathfinding algorithms in a grid world environment. Watch as multiple agents navigate through obstacles to collect goals using different pathfinding strategies.

## Features

- **Multiple Pathfinding Algorithms**
  - A* (default) - Efficient pathfinding using heuristic search
  - Dijkstra's - Guaranteed shortest path finding
  
- **Environment Configurations**
  - Dense - Challenging environment with many goals and obstacles
  - Sparse - More open environment with spread out goals
  
- **Real-time Visualization**
  - Color-coded grid cells (agents, goals, obstacles)
  - Numbered agents for easy tracking
  - Visible pathfinding trails
  - Status panel with live statistics
  - Interactive controls

## Requirements

- Python 3.6+
- Pygame
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/treyisrael7/multi-agent-path-planning
cd path-planning
```

2. Install dependencies:
```bash
pip install pygame numpy
```

## Usage

Run the simulation with different configurations and algorithms:

```bash
# Default (Dense configuration with A*)
python main.py

# Dense environment with Dijkstra's algorithm
python main.py dense dijkstra

# Sparse environment with A*
python main.py sparse astar
```

### Controls

- **SPACE** - Pause/Resume simulation
- **R** - Reset environment
- **ESC** - Quit simulation

### Environment Configurations

1. **Dense/Sparse**
   - Grid: 100x100
   - Agents: 5
   - Goals: 200 (clustered)
   - Obstacles: 30
   - Cell Size: 7px

## Project Structure

```
path_planning/
├── algorithms/
│   └── pathfinding/
│       ├── astar.py
│       └── dijkstra.py
├── env/
│   ├── configurations.py
│   ├── grid_world.py
│   ├── path_manager.py
│   └── movement_manager.py
├── visualization/
│   ├── renderer.py
│   ├── controls.py
│   └── visualizer.py
└── main.py
```

## Visualization Features

- **Grid Display**
  - White: Empty cells
  - Orange: Agents (numbered 1-5)
  - Green: Goals
  - Gray: Obstacles
  - Blue: Pathfinding trails

- **Status Panel**
  - Step counter
  - Goals collected/remaining
  - Time elapsed
  - Control reminders

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
