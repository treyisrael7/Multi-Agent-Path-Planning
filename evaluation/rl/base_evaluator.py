"""Base evaluator for RL agents"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import Dict, List, Tuple

class BaseEvaluator:
    """Base class for RL agent evaluators"""
    
    def __init__(self, 
                 model_path: str,
                 grid_size: int = 50,
                 num_agents: int = 3,
                 num_goals: int = 50,
                 num_obstacles: int = 15,
                 max_steps: int = 1000,
                 visualize: bool = True):
        """Initialize the evaluator
        
        Args:
            model_path: Path to the trained model checkpoint
            grid_size: Size of the grid environment
            num_agents: Number of agents in the environment
            num_goals: Number of goals to collect
            num_obstacles: Number of obstacles in the environment
            max_steps: Maximum steps per episode
            visualize: Whether to show visualization
        """
        self.model_path = model_path
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.visualize = visualize
        
        # Initialize metrics
        self.metrics = {
            'steps': 0,
            'goals_collected': 0,
            'obstacles_hit': 0,
            'total_reward': 0.0
        }
        
        # Visualization setup
        if self.visualize:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.ax.set_xlim(0, grid_size)
            self.ax.set_ylim(0, grid_size)
            self.ax.set_xticks(np.arange(0, grid_size + 1, 5))
            self.ax.set_yticks(np.arange(0, grid_size + 1, 5))
            self.ax.grid(True)
            
            # Initialize visualization elements
            self.obstacle_plot = None
            self.goal_plot = None
            self.agent_plots = []
            self.path_plots = []
            
    def _draw_environment(self, state: np.ndarray) -> None:
        """Draw the current state of the environment
        
        Args:
            state: Current state array [obstacles, agents, goals]
        """
        if not self.visualize:
            return
            
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_xticks(np.arange(0, self.grid_size + 1, 5))
        self.ax.set_yticks(np.arange(0, self.grid_size + 1, 5))
        self.ax.grid(True)
        
        # Draw obstacles
        obstacles = state[0]
        obs_y, obs_x = np.where(obstacles == 1)
        self.ax.scatter(obs_x, obs_y, color='black', marker='s', s=100)
        
        # Draw goals
        goals = state[2]
        goal_y, goal_x = np.where(goals == 1)
        self.ax.scatter(goal_x, goal_y, color='green', marker='s', s=100)
        
        # Draw agents
        agents = state[1]
        agent_y, agent_x = np.where(agents == 1)
        colors = ['red', 'blue', 'purple', 'orange', 'cyan']
        for i, (x, y) in enumerate(zip(agent_x, agent_y)):
            color = colors[i % len(colors)]
            self.ax.scatter(x, y, color=color, marker='o', s=200)
            
        # Update title with metrics
        title = f"Steps: {self.metrics['steps']} | "
        title += f"Goals: {self.metrics['goals_collected']} | "
        title += f"Obstacles: {self.metrics['obstacles_hit']} | "
        title += f"Reward: {self.metrics['total_reward']:.2f}"
        self.ax.set_title(title)
        
        plt.pause(0.01)
        
    def _update_metrics(self, 
                       steps: int,
                       goals_collected: int,
                       obstacles_hit: int,
                       reward: float) -> None:
        """Update evaluation metrics
        
        Args:
            steps: Number of steps taken
            goals_collected: Number of goals collected
            obstacles_hit: Number of obstacles hit
            reward: Total reward received
        """
        self.metrics['steps'] = steps
        self.metrics['goals_collected'] = goals_collected
        self.metrics['obstacles_hit'] = obstacles_hit
        self.metrics['total_reward'] = reward
        
    def evaluate_episode(self) -> Dict[str, float]:
        """Evaluate a single episode
        
        Returns:
            Dictionary of episode metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate_episode")
        
    def evaluate(self, num_episodes: int = 10) -> Dict[str, List[float]]:
        """Evaluate the agent over multiple episodes
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of metrics averaged over episodes
        """
        all_metrics = {
            'steps': [],
            'goals_collected': [],
            'obstacles_hit': [],
            'total_reward': []
        }
        
        for episode in range(num_episodes):
            print(f"\nEvaluating episode {episode + 1}/{num_episodes}")
            episode_metrics = self.evaluate_episode()
            
            for metric, value in episode_metrics.items():
                all_metrics[metric].append(value)
                
        # Calculate averages
        avg_metrics = {
            metric: np.mean(values)
            for metric, values in all_metrics.items()
        }
        
        print("\nAverage metrics over episodes:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.2f}")
            
        return avg_metrics
        
    def plot_results(self, results: Dict[str, Dict[str, float]], save_path: str) -> None:
        """Plot evaluation results
        
        Args:
            results: Dictionary containing evaluation metrics
            save_path: Path to save the plot
        """
        metrics = ['reward', 'steps', 'goals', 'path_length']
        means = [results[metric]['mean'] for metric in metrics]
        stds = [results[metric]['std'] for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, means, yerr=stds, capsize=5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Evaluation Results')
        plt.ylabel('Value')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def print_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print evaluation results
        
        Args:
            results: Dictionary containing evaluation metrics
        """
        print("\nEvaluation Metrics:")
        print("-" * 50)
        for metric, values in results.items():
            print(f"{metric.capitalize()}:")
            print(f"  Mean: {values['mean']:.2f}")
            print(f"  Std: {values['std']:.2f}")
            print() 