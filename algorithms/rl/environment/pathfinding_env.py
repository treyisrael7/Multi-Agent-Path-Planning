"""Reinforcement learning environment for pathfinding"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

class PathfindingEnv(gym.Env):
    """Environment for pathfinding using reinforcement learning"""
    
    def __init__(self, grid_size=50, num_agents=3, num_goals=50, num_obstacles=15, 
                 max_steps=1000, goal_reward=10.0, obstacle_penalty=-1.0, step_penalty=-0.05,
                 progress_reward=0.3, curiosity_reward=0.1):
        """Initialize the environment
        
        Args:
            grid_size: Size of the grid (width and height)
            num_agents: Number of agents in the environment
            num_goals: Number of goals to collect
            num_obstacles: Number of obstacles to place
            max_steps: Maximum number of steps per episode
            goal_reward: Reward for reaching a goal
            obstacle_penalty: Penalty for hitting an obstacle
            step_penalty: Small penalty for each step to encourage efficiency
            progress_reward: Reward for moving closer to any goal
            curiosity_reward: Reward for exploring new areas
        """
        super(PathfindingEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.obstacle_penalty = obstacle_penalty
        self.step_penalty = step_penalty
        self.progress_reward = progress_reward
        self.curiosity_reward = curiosity_reward
        
        # Action space: 8 possible movements per agent
        self.action_space = spaces.MultiDiscrete([8] * num_agents)
        
        # Observation space: grid state + agent positions + goal positions
        # 3 channels: obstacles, agent positions, goal positions
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 3),
            dtype=np.float32
        )
        
        # Movement directions (8-way movement)
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Top row
            (0, -1),           (0, 1),   # Middle row
            (1, -1),  (1, 0),  (1, 1)    # Bottom row
        ]
        
        # Metrics
        self.goals_collected = 0
        self.obstacles_hit = 0
        self.steps_taken = 0
        
        # For tracking visited cells for curiosity reward
        self.visited_cells = set()
        
        # For rendering
        self.fig = None
        self.ax = None
        
        # Reward parameters
        self.episode_count = 0  # Track episodes for decay
        
        self.reset()
        
    def reset(self):
        """Reset the environment
        
        Returns:
            Initial observation
        """
        # Create empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Place obstacles randomly
        obstacle_positions = []
        while len(obstacle_positions) < self.num_obstacles:
            pos = (np.random.randint(0, self.grid_size), 
                  np.random.randint(0, self.grid_size))
            if pos not in obstacle_positions:
                obstacle_positions.append(pos)
                self.grid[pos] = 1
        
        # Place goals randomly
        self.goals = []
        while len(self.goals) < self.num_goals:
            pos = (np.random.randint(0, self.grid_size), 
                  np.random.randint(0, self.grid_size))
            if pos not in obstacle_positions and pos not in self.goals:
                self.goals.append(pos)
        
        # Place agents randomly
        self.agent_positions = []
        while len(self.agent_positions) < self.num_agents:
            pos = (np.random.randint(0, self.grid_size), 
                  np.random.randint(0, self.grid_size))
            if (pos not in obstacle_positions and 
                pos not in self.goals and 
                pos not in self.agent_positions):
                self.agent_positions.append(pos)
        
        # Initialize step counter
        self.steps = 0
        
        # Reset metrics for this episode
        self.episode_goals = 0
        self.episode_obstacles = 0
        
        # Reset visited cells
        self.visited_cells = set()
        for pos in self.agent_positions:
            self.visited_cells.add(pos)
        
        self.episode_count += 1
        return self._get_observation()
        
    def step(self, actions):
        """Take a step in the environment
        
        Args:
            actions: List of actions for each agent (0-7 for 8 directions)
            
        Returns:
            observation: New observation
            rewards: List of rewards for each agent
            done: Whether episode is done
            info: Additional information
        """
        self.steps += 1
        self.steps_taken += 1
        
        # Decay exploration rewards over time
        decay_factor = max(0.1, 1.0 - (self.episode_count / 500))  # Decay over 500 episodes
        current_curiosity = self.curiosity_reward * decay_factor
        current_progress = self.progress_reward * decay_factor
        
        # Initialize rewards for each agent
        rewards = [0.0] * self.num_agents
        done = False
        
        # Move each agent independently
        for i, action in enumerate(actions):
            # Store previous position
            prev_pos = self.agent_positions[i]
            
            # Get movement direction
            dx, dy = self.directions[action]
            
            # Calculate new position
            new_x = self.agent_positions[i][0] + dx
            new_y = self.agent_positions[i][1] + dy
            new_pos = (new_x, new_y)
            
            # Check if move is valid
            if (0 <= new_x < self.grid_size and 
                0 <= new_y < self.grid_size):
                
                # Check if hit obstacle
                if self.grid[new_pos]:
                    self.obstacles_hit += 1
                    self.episode_obstacles += 1
                    rewards[i] += self.obstacle_penalty
                else:
                    # Valid move
                    self.agent_positions[i] = new_pos
                    
                    # Check if reached goal
                    if new_pos in self.goals:
                        self.goals_collected += 1
                        self.episode_goals += 1
                        self.goals.remove(new_pos)
                        rewards[i] += self.goal_reward
                        
                        # Check if all goals collected
                        if len(self.goals) == 0:
                            done = True
                    else:
                        # Calculate progress reward
                        min_prev_dist = min([abs(prev_pos[0] - g[0]) + abs(prev_pos[1] - g[1]) for g in self.goals])
                        min_new_dist = min([abs(new_pos[0] - g[0]) + abs(new_pos[1] - g[1]) for g in self.goals])
                        if min_new_dist < min_prev_dist:
                            rewards[i] += current_progress
                        
                        # Add curiosity reward for exploring new cells
                        if new_pos not in self.visited_cells:
                            rewards[i] += current_curiosity
                            self.visited_cells.add(new_pos)
                        
                        # Small negative reward for each step
                        rewards[i] += self.step_penalty
            else:
                # Out of bounds
                rewards[i] += self.obstacle_penalty
        
        # Check if episode is done
        if self.steps >= self.max_steps:
            done = True
            
        # Additional info
        info = {
            'goals_collected': self.episode_goals,
            'obstacles_hit': self.episode_obstacles,
            'steps': self.steps,
            'goals_remaining': len(self.goals),
            'rewards': rewards
        }
        
        # Add bonus for finishing early
        if len(self.goals) == 0:
            remaining_steps = self.max_steps - self.steps
            early_completion_bonus = remaining_steps * 0.01
            for i in range(self.num_agents):
                rewards[i] += early_completion_bonus
            
        return self._get_observation(), rewards, done, info
        
    def _get_observation(self):
        """Get the current state observation"""
        # Create observation array with shape (H, W, C)
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Channel 0: Obstacles
        obs[:, :, 0] = self.grid
        
        # Channel 1: Current positions of all agents
        for pos in self.agent_positions:
            obs[pos[0], pos[1], 1] = 1
            
        # Channel 2: Goals
        for goal in self.goals:
            obs[goal[0], goal[1], 2] = 1
            
        # Transpose to (C, H, W) for PyTorch
        obs = np.transpose(obs, (2, 0, 1))
        
        return obs
        
    def render(self, mode='human'):
        """Render the environment
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
                plt.ion()  # Turn on interactive mode
            
            self.ax.clear()
            
            # Set up the grid
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_xticks(np.arange(0, self.grid_size + 1, 10))
            self.ax.set_yticks(np.arange(0, self.grid_size + 1, 10))
            self.ax.grid(True, linestyle='-', alpha=0.7)
            
            # Draw obstacles
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.grid[x, y] == 1:
                        self.ax.add_patch(Rectangle((x, y), 1, 1, facecolor='black', alpha=0.5))
            
            # Draw goals
            for goal in self.goals:
                self.ax.add_patch(Rectangle((goal[0], goal[1]), 1, 1, facecolor='red', alpha=0.7))
            
            # Draw agents
            for i, agent in enumerate(self.agent_positions):
                color = 'blue' if i == 0 else 'green'  # First agent is blue, others are green
                self.ax.add_patch(Rectangle((agent[0], agent[1]), 1, 1, facecolor=color, alpha=0.7))
            
            # Add title with metrics
            self.ax.set_title(f'Step: {self.steps}, Goals: {self.episode_goals}/{self.num_goals}, Obstacles: {self.episode_obstacles}')
            
            # Draw and pause
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
            
    def close(self):
        """Close the environment"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            
    def seed(self, seed=None):
        """Set the random seed
        
        Args:
            seed: Random seed
        """
        np.random.seed(seed)
        return [seed]
        
    def get_metrics(self):
        """Get environment metrics
        
        Returns:
            Dictionary of metrics
        """
        return {
            'goals_collected': self.goals_collected,
            'obstacles_hit': self.obstacles_hit,
            'steps_taken': self.steps_taken,
            'goals_remaining': len(self.goals)
        }
        
    def get_state(self, for_rl=True):
        """Get the current state
        
        Args:
            for_rl: Whether to return state in format for RL agents
            
        Returns:
            State representation
        """
        if for_rl:
            return self._get_observation()
        else:
            # Return state in format for A* and Dijkstra
            return {
                'grid': self.grid,
                'start': self.agent_positions[0],
                'goal': self.goals[0] if self.goals else None
            } 