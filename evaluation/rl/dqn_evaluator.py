"""DQN evaluator implementation"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import os
import time
import pygame

from algorithms.rl.environment.pathfinding_env import PathfindingEnv
from algorithms.rl.agents.dqn import DQNAgent
from evaluation.rl.base_evaluator import BaseEvaluator

class DQNEvaluator(BaseEvaluator):
    """Evaluator for DQN agent"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 grid_size: int = 50,
                 num_agents: int = 3,
                 num_goals: int = 100,
                 num_obstacles: int = 15,
                 max_steps: int = 1000,
                 visualize: bool = True):
        """Initialize DQN evaluator
        
        Args:
            model_path: Path to the trained model checkpoint
            grid_size: Size of the grid environment
            num_agents: Number of agents in the environment
            num_goals: Number of goals to collect
            num_obstacles: Number of obstacles in the environment
            max_steps: Maximum steps per episode
            visualize: Whether to show visualization
        """
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join('models', 'dqn', 'dqn_model.pth')
            
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
            
        super().__init__(
            model_path=model_path,
            grid_size=grid_size,
            num_agents=num_agents,
            num_goals=num_goals,
            num_obstacles=num_obstacles,
            max_steps=max_steps,
            visualize=visualize
        )
        
        # Initialize environment
        self.env = PathfindingEnv(
            grid_size=grid_size,
            num_agents=num_agents,
            num_goals=num_goals,
            num_obstacles=num_obstacles,
            max_steps=max_steps
        )
        
        # Initialize agent with correct state shape
        # The environment returns state in [H, W, C] format
        # But the DQN expects [C, H, W] format
        state_shape = (3, grid_size, grid_size)  # (channels, height, width)
        action_size = self.env.action_space.nvec[0]  # Get action size for one agent
        self.agent = DQNAgent(
            state_shape=state_shape,
            action_size=action_size,
            num_agents=num_agents
        )
        
        # Load model
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the trained model from checkpoint"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Print model information
            print("\nLoading model from checkpoint:")
            print(f"State shape: {self.agent.state_shape}")
            print(f"Action size: {self.agent.action_size}")
            print(f"Number of agents: {self.agent.num_agents}")
            
            # Load each policy network and set to evaluation mode
            for i in range(self.num_agents):
                self.agent.policy_nets[i].load_state_dict(checkpoint['policy_nets'][i])
                self.agent.policy_nets[i].eval()
                
            print(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Convert state to correct format for DQN agent
        
        Args:
            state: State array from environment [H, W, C] format
            
        Returns:
            Processed state array in [C, H, W] format
        """
        # Convert from [H, W, C] to [C, H, W] format
        if state.shape[-1] == 3:  # If last dimension is channels
            state = np.transpose(state, (2, 0, 1))
            
        # Ensure state is float32
        if state.dtype != np.float32:
            state = state.astype(np.float32)
            
        return state
            
    def evaluate_episode(self) -> Dict[str, float]:
        """Evaluate a single episode
        
        Returns:
            Dictionary of episode metrics
        """
        state = self.env.reset()
        done = False
        steps = 0
        goals_collected = 0
        obstacles_hit = 0
        total_reward = 0.0
        
        while not done and steps < self.max_steps:
            # Select actions using agent's method (same as training)
            with torch.no_grad():
                actions, q_values = self.agent.select_action(state, evaluate=False)  # emulate training behavior
            
            # Convert actions to numpy array
            actions = np.array(actions)
            
            # Take step in environment
            next_state, rewards, done, info = self.env.step(actions)
            
            # Update metrics
            steps += 1
            goals_collected = info['goals_collected']
            obstacles_hit = info['obstacles_hit']
            total_reward += sum(rewards)
            
            # Update visualization
            if self.visualize:
                self._update_metrics(steps, goals_collected, obstacles_hit, total_reward)
                self._draw_environment(state)
                
                # Handle events and timing
                self.clock.tick(self.target_fps * self.speed)
                
                # Check for events
                result = self._handle_events()
                if result is False:  # Quit signal
                    return {
                        'steps': steps,
                        'goals_collected': goals_collected,
                        'obstacles_hit': obstacles_hit,
                        'total_reward': total_reward
                    }
                elif result is True:  # Reset signal
                    # Reset environment and start new episode
                    state = self.env.reset()
                    steps = 0
                    goals_collected = 0
                    obstacles_hit = 0
                    total_reward = 0.0
                    continue
                
                # Handle pause
                while self.paused and self.running:
                    result = self._handle_events()
                    if result is False:  # Quit signal
                        return {
                            'steps': steps,
                            'goals_collected': goals_collected,
                            'obstacles_hit': obstacles_hit,
                            'total_reward': total_reward
                        }
                    elif result is True:  # Reset signal
                        # Reset environment and start new episode
                        state = self.env.reset()
                        steps = 0
                        goals_collected = 0
                        obstacles_hit = 0
                        total_reward = 0.0
                        continue
                    self.clock.tick(self.target_fps * self.speed)
                
            # Update state for next iteration
            state = next_state
            
        # Print episode summary
        print(f"\nEpisode complete - Steps: {steps}, Goals: {goals_collected}, Obstacles: {obstacles_hit}, Reward: {total_reward:.2f}")
            
        return {
            'steps': steps,
            'goals_collected': goals_collected,
            'obstacles_hit': obstacles_hit,
            'total_reward': total_reward
        }
        
def main():
    """Main function to run DQN evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate DQN agent')
    parser.add_argument('--model_path', type=str,
                      help='Path to the trained model checkpoint (default: models/dqn/dqn_model.pth)')
    parser.add_argument('--no-vis', action='store_true',
                      help='Disable visualization')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate (default: 10)')
    
    args = parser.parse_args()
    
    evaluator = DQNEvaluator(
        model_path=args.model_path,
        visualize=not args.no_vis
    )
    
    evaluator.evaluate(num_episodes=args.episodes)
    
if __name__ == '__main__':
    main() 