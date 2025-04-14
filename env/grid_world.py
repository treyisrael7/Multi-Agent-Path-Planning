"""Grid world environment for pathfinding"""

import numpy as np
from algorithms.pathfinding import astar_path
from env.configurations import BaseConfig
from env.path_manager import PathManager
from env.movement_manager import MovementManager

class GridWorld:
    """Grid world environment for pathfinding"""
    
    def __init__(self, config=None, pathfinding_algorithm=None):
        if config is None:
            config = BaseConfig()
        
        self.config = config
        self.size = config.size
        self.num_agents = config.num_agents
        self.num_goals = config.num_goals
        self.num_obstacles = config.num_obstacles
        self.grid = np.zeros((self.size, self.size), dtype=int)  # 0: empty, 1: obstacle, 2: goal, 3: agent
        self.agent_positions = []
        self.goal_positions = []
        
        # Initialize managers
        self.path_manager = PathManager(self.size, pathfinding_algorithm if pathfinding_algorithm else astar_path)
        self.movement_manager = MovementManager(self.size)
        
        self.reset()
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions"""
        return self.movement_manager.get_neighbors(pos, self.grid)
    
    def get_state(self):
        """Get the current state of the environment"""
        # Extract just the paths from the (path, metrics) tuples
        paths = [path for path, _ in self.path_manager.paths] if self.path_manager.paths else []
        
        return {
            'grid': self.grid.copy(),
            'agents': self.agent_positions.copy(),
            'goals': self.goal_positions.copy(),
            'paths': [path.copy() if path else [] for path in paths]
        }
    
    def get_valid_actions(self, agent_idx):
        """Get valid actions for an agent"""
        return self.movement_manager.get_valid_actions(self.agent_positions[agent_idx], self.grid)
    
    def step(self, actions=None):
        """Step the environment forward"""
        if actions is None:
            # Path-based movement
            moves, goals_collected = self.movement_manager.execute_moves(
                self.grid, 
                self.agent_positions, 
                self.path_manager.paths,
                self.goal_positions
            )
            
            # Update paths if goals were collected
            if goals_collected > 0:
                self.path_manager.clear_cache()
                self.path_manager.update_all_paths(
                    self.agent_positions,
                    self.goal_positions,
                    self.get_neighbors
                )
        else:
            # Action-based movement
            moves, goals_collected = self.movement_manager.execute_direct_moves(
                self.grid,
                self.agent_positions,
                actions,
                self.goal_positions
            )
        
        # Check if episode is done
        done = len(self.goal_positions) == 0
        
        # Return the expected 5 values
        return self.grid, self.agent_positions, self.goal_positions, done, {'moves': moves, 'goals_collected': goals_collected}
    
    def reset(self):
        """Reset the environment with new random positions"""
        self.grid.fill(0)
        self.agent_positions = []
        self.goal_positions = []
        self.path_manager.initialize(self.num_agents)
        
        # Place obstacles according to configuration pattern
        pattern = self.config.get_obstacle_pattern()
        for _ in range(self.num_obstacles):
            while True:
                x = np.random.randint(0, self.size - max(p[0] for p in pattern) - 1)
                y = np.random.randint(0, self.size - max(p[1] for p in pattern) - 1)
                valid = True
                
                # Check if pattern fits
                for dx, dy in pattern:
                    if self.grid[x + dx, y + dy] != 0:
                        valid = False
                        break
                
                if valid:
                    for dx, dy in pattern:
                        self.grid[x + dx, y + dy] = 1
                    break
        
        # Place agents
        for _ in range(self.num_agents):
            while True:
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = 3
                    self.agent_positions.append((x, y))
                    break
        
        # Place goals according to configuration strategy
        strategy = self.config.get_goal_placement_strategy()
        goals_placed = 0
        attempts = 0
        
        while goals_placed < self.num_goals and attempts < strategy['max_attempts']:
            # Place a cluster of goals
            cluster_size = min(strategy['cluster_size'], self.num_goals - goals_placed)
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            
            if self.grid[x, y] == 0:
                self.grid[x, y] = 2
                self.goal_positions.append((x, y))
                goals_placed += 1
                
                # Try to place remaining goals in cluster
                for _ in range(cluster_size - 1):
                    spread = strategy['spread']
                    dx = np.random.randint(-spread, spread + 1)
                    dy = np.random.randint(-spread, spread + 1)
                    new_x, new_y = x + dx, y + dy
                    
                    if (0 <= new_x < self.size and 
                        0 <= new_y < self.size and 
                        self.grid[new_x, new_y] == 0):
                        self.grid[new_x, new_y] = 2
                        self.goal_positions.append((new_x, new_y))
                        goals_placed += 1
            
            attempts += 1
        
        # Update paths for all agents
        self.path_manager.clear_cache()
        self.path_manager.update_all_paths(
            self.agent_positions,
            self.goal_positions,
            self.get_neighbors
        )