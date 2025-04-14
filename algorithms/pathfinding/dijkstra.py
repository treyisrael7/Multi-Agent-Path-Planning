"""Dijkstra's pathfinding algorithm implementation"""

import heapq

def dijkstra_path(start, goal, get_neighbors, grid_size):
    """Find path using Dijkstra's algorithm
    
    Returns:
        tuple: (path, metrics) where path is list of positions and metrics is a dict
    """
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    
    # Priority queue of (distance, position)
    open_set = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}  # Cost from start to current position
    
    # Track metrics
    nodes_expanded = 0
    nodes_visited = set()
    
    while open_set:
        current_dist, current = heapq.heappop(open_set)
        nodes_visited.add(current)
        nodes_expanded += 1
        
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            
            metrics = {
                'nodes_expanded': nodes_expanded,
                'nodes_visited': len(nodes_visited)
            }
            return path[::-1], metrics
        
        for next_pos in get_neighbors(current):
            next_pos = tuple(map(int, next_pos))
            tentative_g = g_score[current] + 1
            
            if next_pos not in g_score or tentative_g < g_score[next_pos]:
                came_from[next_pos] = current
                g_score[next_pos] = tentative_g
                heapq.heappush(open_set, (tentative_g, next_pos))
                nodes_visited.add(next_pos)
    
    # No path found
    metrics = {
        'nodes_expanded': nodes_expanded,
        'nodes_visited': len(nodes_visited)
    }
    return [], metrics 