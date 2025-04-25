"""Path management for the grid world environment"""

import numpy as np
from algorithms.search.astar import manhattan_distance

class PathManager:
    """Manages paths and path caching for agents in the grid world"""
    
    def __init__(self, grid_size, pathfinding_algorithm):
        self.size = grid_size
        self.pathfinding_algorithm = pathfinding_algorithm
        self.path_cache = {}
        self.paths = []
        
    def initialize(self, num_agents):
        """Initialize paths for the given number of agents"""
        self.paths = [[] for _ in range(num_agents)]
        
    def get_cached_path(self, start, goal, get_neighbors):
        """Get a cached path or compute and cache a new one"""
        key = (start, goal)
        if key in self.path_cache:
            return self.path_cache[key]
        
        path = self.pathfinding_algorithm(start, goal, get_neighbors, self.size)
        if path:
            self.path_cache[key] = path
        return path
        
    def clear_cache(self):
        """Clear the path cache when the environment changes"""
        self.path_cache = {}
        
    def update_all_paths(self, agent_positions, goal_positions, get_neighbors):
        """Update paths for all agents"""
        if not goal_positions:
            self.paths = [[] for _ in range(len(agent_positions))]
            return
        
        # Quick assignment using manhattan distance first
        assignments = []  # List of (distance, agent_idx, goal) tuples
        for i, agent_pos in enumerate(agent_positions):
            for goal in goal_positions:
                dist = manhattan_distance(agent_pos, goal)
                assignments.append((dist, i, goal))
        
        assignments.sort()  # Sort by distance
        assigned_goals = set()
        self.paths = [[] for _ in range(len(agent_positions))]
        
        # Assign goals to agents
        for dist, agent_idx, goal in assignments:
            if goal in assigned_goals or self.paths[agent_idx]:
                continue
            
            path = self.get_cached_path(agent_positions[agent_idx], goal, get_neighbors)
            if path:
                self.paths[agent_idx] = path
                assigned_goals.add(goal)
        
        # Assign any remaining agents to closest available goals
        for i, path in enumerate(self.paths):
            if not path and goal_positions:
                closest_goal = min(goal_positions, 
                                 key=lambda g: manhattan_distance(agent_positions[i], g))
                path = self.get_cached_path(agent_positions[i], closest_goal, get_neighbors)
                if path:
                    self.paths[i] = path
                    
    def get_path(self, agent_idx):
        """Get the current path for an agent"""
        return self.paths[agent_idx] if agent_idx < len(self.paths) else []
    
    def update_path(self, agent_idx, new_path):
        """Update the path for an agent"""
        if agent_idx < len(self.paths):
            self.paths[agent_idx] = new_path 