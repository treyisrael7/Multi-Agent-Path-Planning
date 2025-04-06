"""Different environment configurations for the grid world"""

from algorithms.pathfinding import dijkstra_path

class BaseConfig:
    """Base configuration class"""
    def __init__(self):
        self.size = 40
        self.num_agents = 5
        self.num_goals = 30
        self.num_obstacles = 20
        self.cell_size = 15
        
    def get_obstacle_pattern(self):
        """Single obstacle blocks"""
        return [(0,0)]
        
    def get_goal_placement_strategy(self):
        """Default goal placement strategy"""
        return {
            'cluster_size': 2,
            'spread': 4,
            'max_attempts': 50
        }

class DenseConfig(BaseConfig):
    """Dense environment with many clustered goals"""
    def __init__(self):
        super().__init__()
        self.size = 100
        self.num_agents = 5
        self.num_goals = 200
        self.num_obstacles = 30
        self.cell_size = 7
        
    def get_obstacle_pattern(self):
        """2x2 obstacle blocks"""
        return [(0,0), (0,1), (1,0), (1,1)]
        
    def get_goal_placement_strategy(self):
        """Dense clustered goal placement"""
        return {
            'cluster_size': 8,
            'spread': 6,
            'max_attempts': 25
        }

class SparseConfig(BaseConfig):
    """Sparse environment with spread out goals"""
    def __init__(self):
        super().__init__()
        self.size = 100
        self.num_agents = 5
        self.num_goals = 200
        self.num_obstacles = 30
        self.cell_size = 7
        
    def get_obstacle_pattern(self):
        """2x2 obstacle blocks"""
        return [(0,0), (0,1), (1,0), (1,1)]
        
    def get_goal_placement_strategy(self):
        """Sparse spread out goal placement"""
        return {
            'cluster_size': 2,
            'spread': 8,
            'max_attempts': 50
        }

# Dictionary of available configurations
CONFIGS = {
    'dense': DenseConfig,
    'sparse': SparseConfig,
    'default': DenseConfig
}   