# Path Planning Project

This project implements both traditional pathfinding algorithms and reinforcement learning approaches for path planning in a grid world environment.

## Project Structure

- `algorithms/`: Contains pathfinding and RL algorithms
  - `pathfinding/`: Traditional pathfinding algorithms (A*, Dijkstra)
  - `rl/`: Reinforcement learning algorithms (DQN, A2C)
- `env/`: Environment implementations
  - `grid_world.py`: Grid world environment for pathfinding
  - `movement_manager.py`: Manages agent movements
  - `path_manager.py`: Manages paths for agents
  - `configurations.py`: Environment configurations

## Pathfinding Algorithms

The project implements traditional pathfinding algorithms:
- A* Algorithm
- Dijkstra's Algorithm

These algorithms are used in the `GridWorld` environment to find optimal paths for agents to reach goals while avoiding obstacles.

## Reinforcement Learning

The project also implements reinforcement learning approaches:
- Deep Q-Network (DQN)
- Advantage Actor-Critic (A2C)

These algorithms learn to navigate the environment through trial and error, with the goal of maximizing rewards by collecting goals while avoiding obstacles.

## Environment

The project includes two environment implementations:
1. `GridWorld`: A multi-agent environment for pathfinding algorithms
2. `PathfindingEnv`: A Gym-compatible environment for reinforcement learning

## Usage

### Pathfinding

```python
from env.grid_world import GridWorld
from env.configurations import DenseConfig

# Create environment with dense configuration
config = DenseConfig()
env = GridWorld(config=config)

# Reset environment
env.reset()

# Get valid actions for an agent
actions = env.get_valid_actions(0)

# Step environment
grid, agent_positions, goal_positions, done, info = env.step(actions)
```

### Reinforcement Learning

```python
from algorithms.rl.environment import PathfindingEnv
from algorithms.rl.dqn import DQNAgent

# Create environment
env = PathfindingEnv(grid_size=50, obstacle_density=0.3)

# Create agent
agent = DQNAgent(
    state_shape=env.observation_space.shape,
    action_size=env.action_space.n
)

# Training loop
state = env.reset()
done = False
while not done:
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    agent.store_transition(state, action, reward, next_state, done)
    state = next_state
```

## Training

To train RL agents:

```bash
# Train DQN agent
python algorithms/rl/train.py --agent dqn --episodes 1000 --grid_size 50 --obstacle_density 0.3

# Train A2C agent
python algorithms/rl/train.py --agent a2c --episodes 1000 --grid_size 50 --obstacle_density 0.3

# Train both agents
python algorithms/rl/train.py --agent both --episodes 1000 --grid_size 50 --obstacle_density 0.3
```

## Requirements

See `requirements.txt` for dependencies.
