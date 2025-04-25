"""Movement management for the grid world environment"""

import numpy as np

class MovementManager:
    """Manages agent movement and collision avoidance in the grid world"""
    
    def __init__(self, grid_size):
        self.size = grid_size
        
    def get_valid_actions(self, pos, grid):
        """Get valid actions for an agent at the given position"""
        x, y = pos
        valid_actions = []
        
        # Check each direction (up, down, left, right)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.size and 
                0 <= new_y < self.size and 
                grid[new_x, new_y] != 1):  # Not an obstacle
                valid_actions.append((new_x, new_y))
        
        return valid_actions
        
    def get_neighbors(self, pos, grid):
        """Get valid neighboring positions"""
        return self.get_valid_actions(pos, grid)
        
    def execute_moves(self, grid, agent_positions, paths, goal_positions):
        """Execute moves for all agents using path-based movement"""
        moves = []
        goals_collected = 0
        current_positions = {tuple(pos) for pos in agent_positions}
        planned_moves = {}
        
        # Sort agents by path length
        agents_to_move = [(len(path[0]), i) for i, (path, _) in enumerate(paths) if path and len(path) > 1]
        agents_to_move.sort()
        
        # Execute moves
        for _, i in agents_to_move:
            path, _ = paths[i]
            next_pos = tuple(path[1])
            
            if next_pos not in planned_moves and next_pos not in current_positions:
                # Move is valid
                agent_pos = tuple(agent_positions[i])
                grid[agent_pos] = 0
                grid[next_pos] = 3
                agent_positions[i] = next_pos
                paths[i] = (path[1:], paths[i][1])  # Keep the metrics
                planned_moves[next_pos] = i
                moves.append((i, next_pos))
                
                # Check for goal collection
                if next_pos in goal_positions:
                    goal_positions.remove(next_pos)
                    goals_collected += 1
                    
        return moves, goals_collected
        
    def execute_direct_moves(self, grid, agent_positions, actions, goal_positions):
        """Execute moves for all agents using direct action-based movement"""
        moves = []
        goals_collected = 0
        current_positions = set(agent_positions)
        
        for i, new_pos in enumerate(actions):
            if new_pos not in current_positions:
                old_pos = agent_positions[i]
                grid[old_pos] = 0
                grid[new_pos] = 3
                agent_positions[i] = new_pos
                moves.append((i, new_pos))
                current_positions.add(new_pos)
                
                # Check for goal collection
                if new_pos in goal_positions:
                    goal_positions.remove(new_pos)
                    goals_collected += 1
                    
        return moves, goals_collected 