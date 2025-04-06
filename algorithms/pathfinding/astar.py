"""A* pathfinding algorithm implementation"""

import heapq
import numpy as np

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two points"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def astar_path(start, goal, get_neighbors, grid_size):
    """Find path using A* algorithm"""
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    
    # Priority queue of (f_score, position)
    open_set = [(manhattan_distance(start, goal), start)]
    came_from = {start: None}
    g_score = {start: 0}  # Cost from start to current position
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for next_pos in get_neighbors(current):
            next_pos = tuple(map(int, next_pos))
            tentative_g = g_score[current] + 1
            
            if next_pos not in g_score or tentative_g < g_score[next_pos]:
                came_from[next_pos] = current
                g_score[next_pos] = tentative_g
                f_score = tentative_g + manhattan_distance(next_pos, goal)
                heapq.heappush(open_set, (f_score, next_pos))
    
    return []  # No path found 