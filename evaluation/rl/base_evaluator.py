"""Base evaluator for RL agents"""

import numpy as np
import pygame
import time
from typing import Dict, List, Tuple

class BaseEvaluator:
    """Base class for RL agent evaluators"""
    
    def __init__(self, 
                 model_path: str,
                 grid_size: int = 50,
                 num_agents: int = 3,
                 num_goals: int = 100,
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
            pygame.init()
            self.cell_size = 10  # pixels per cell
            self.sidebar_width = 200
            self.window_width = self.grid_size * self.cell_size + self.sidebar_width
            self.window_height = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("RL Agent Evaluation")
            
            # Control variables
            self.paused = False
            self.running = True
            self.speed = 0.1  # Much slower default speed
            self.show_paths = True
            self.clock = pygame.time.Clock()
            self.target_fps = 30  # Target frame rate
            
            # Colors
            self.colors = {
                'background': (255, 255, 255),  # White
                'grid': (200, 200, 200),        # Light gray
                'obstacle': (100, 100, 100),    # Dark gray
                'goal': (0, 255, 0),            # Green
                'agent': [(255, 100, 0), (0, 0, 255), (128, 0, 128), (255, 165, 0), (0, 255, 255)],  # Orange, Blue, Purple, Orange, Cyan
                'text': (0, 0, 0),              # Black text
                'button': (180, 180, 180),      # Light gray
                'button_hover': (150, 150, 150),# Darker gray
                'sidebar': (240, 240, 240)      # Light gray
            }
            
            # Button definitions
            self.buttons = {
                'pause': pygame.Rect(self.grid_size * self.cell_size + 20, 20, 160, 30),
                'reset': pygame.Rect(self.grid_size * self.cell_size + 20, 60, 160, 30),
                'speed_up': pygame.Rect(self.grid_size * self.cell_size + 20, 100, 75, 30),
                'speed_down': pygame.Rect(self.grid_size * self.cell_size + 105, 100, 75, 30),
                'toggle_paths': pygame.Rect(self.grid_size * self.cell_size + 20, 140, 160, 30)
            }
            
    def _draw_sidebar(self) -> None:
        """Draw the control sidebar"""
        if not self.visualize:
            return
            
        # Draw sidebar background
        sidebar_rect = pygame.Rect(self.grid_size * self.cell_size, 0, self.sidebar_width, self.window_height)
        pygame.draw.rect(self.screen, self.colors['sidebar'], sidebar_rect)
        
        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        font = pygame.font.Font(None, 24)
        
        # Pause/Play button
        color = self.colors['button_hover'] if self.buttons['pause'].collidepoint(mouse_pos) else self.colors['button']
        pygame.draw.rect(self.screen, color, self.buttons['pause'])
        text = font.render("Pause" if not self.paused else "Play", True, self.colors['text'])
        self.screen.blit(text, (self.buttons['pause'].centerx - text.get_width()//2, 
                               self.buttons['pause'].centery - text.get_height()//2))
        
        # Reset button
        color = self.colors['button_hover'] if self.buttons['reset'].collidepoint(mouse_pos) else self.colors['button']
        pygame.draw.rect(self.screen, color, self.buttons['reset'])
        text = font.render("Reset", True, self.colors['text'])
        self.screen.blit(text, (self.buttons['reset'].centerx - text.get_width()//2,
                               self.buttons['reset'].centery - text.get_height()//2))
        
        # Speed controls
        color = self.colors['button_hover'] if self.buttons['speed_down'].collidepoint(mouse_pos) else self.colors['button']
        pygame.draw.rect(self.screen, color, self.buttons['speed_down'])
        text = font.render("-", True, self.colors['text'])
        self.screen.blit(text, (self.buttons['speed_down'].centerx - text.get_width()//2,
                               self.buttons['speed_down'].centery - text.get_height()//2))
        
        color = self.colors['button_hover'] if self.buttons['speed_up'].collidepoint(mouse_pos) else self.colors['button']
        pygame.draw.rect(self.screen, color, self.buttons['speed_up'])
        text = font.render("+", True, self.colors['text'])
        self.screen.blit(text, (self.buttons['speed_up'].centerx - text.get_width()//2,
                               self.buttons['speed_up'].centery - text.get_height()//2))
        
        # Toggle paths button
        color = self.colors['button_hover'] if self.buttons['toggle_paths'].collidepoint(mouse_pos) else self.colors['button']
        pygame.draw.rect(self.screen, color, self.buttons['toggle_paths'])
        text = font.render("Toggle Paths", True, self.colors['text'])
        self.screen.blit(text, (self.buttons['toggle_paths'].centerx - text.get_width()//2,
                               self.buttons['toggle_paths'].centery - text.get_height()//2))
        
        # Display metrics
        y_offset = 180
        for metric, value in self.metrics.items():
            text = font.render(f"{metric.replace('_', ' ').title()}: {value:.2f}", True, self.colors['text'])
            self.screen.blit(text, (self.grid_size * self.cell_size + 20, y_offset))
            y_offset += 30
            
        # Display speed
        text = font.render(f"Speed: {self.speed:.1f}x", True, self.colors['text'])
        self.screen.blit(text, (self.grid_size * self.cell_size + 20, y_offset))
            
    def _handle_events(self) -> bool:
        """Handle pygame events
        
        Returns:
            bool: Whether to continue running
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Check button clicks
                if self.buttons['pause'].collidepoint(mouse_pos):
                    self.paused = not self.paused
                elif self.buttons['reset'].collidepoint(mouse_pos):
                    return True  # Signal for reset
                elif self.buttons['speed_up'].collidepoint(mouse_pos):
                    self.speed = min(4.0, self.speed + 0.5)
                elif self.buttons['speed_down'].collidepoint(mouse_pos):
                    self.speed = max(0.1, self.speed - 0.5)
                elif self.buttons['toggle_paths'].collidepoint(mouse_pos):
                    self.show_paths = not self.show_paths
                    
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    return True  # Signal for reset
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                    
        return None  # No special action needed
            
    def _draw_environment(self, state: np.ndarray) -> None:
        """Draw the current state of the environment
        
        Args:
            state: Current state array [obstacles, agents, goals]
        """
        if not self.visualize:
            return
            
        # Clear screen with white background
        self.screen.fill(self.colors['background'])
        
        # Draw grid lines
        for i in range(0, self.grid_size * self.cell_size + 1, self.cell_size):
            # Vertical lines
            pygame.draw.line(
                self.screen, 
                self.colors['grid'],
                (i, 0),
                (i, self.grid_size * self.cell_size)
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                self.colors['grid'],
                (0, i),
                (self.grid_size * self.cell_size, i)
            )
        
        # Draw obstacles
        obstacles = state[0]
        obs_y, obs_x = np.where(obstacles == 1)
        for x, y in zip(obs_x, obs_y):
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors['obstacle'], rect)
        
        # Draw goals
        goals = state[2]
        goal_y, goal_x = np.where(goals == 1)
        for x, y in zip(goal_x, goal_y):
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors['goal'], rect)
        
        # Draw agents
        agents = state[1]
        agent_y, agent_x = np.where(agents == 1)
        for i, (x, y) in enumerate(zip(agent_x, agent_y)):
            color = self.colors['agent'][i % len(self.colors['agent'])]
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, rect)
            
            # Draw agent number
            font = pygame.font.Font(None, self.cell_size)
            text = font.render(str(i + 1), True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
            
        # Draw sidebar
        self._draw_sidebar()
        
        # Update display
        pygame.display.flip()
        
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
        
        episode = 0
        while episode < num_episodes and (not self.visualize or self.running):
            print(f"\nEvaluating episode {episode + 1}/{num_episodes}")
            
            # Run episode
            episode_metrics = self.evaluate_episode()
            
            # Store metrics
            for metric, value in episode_metrics.items():
                all_metrics[metric].append(value)
                
            episode += 1
            
            # Handle visualization delay based on speed
            if self.visualize:
                # Wait for user input between episodes
                waiting = True
                while waiting and self.running:
                    result = self._handle_events()
                    if result is False:  # Quit signal
                        return all_metrics
                    elif result is True:  # Reset signal
                        episode -= 1  # Decrement episode counter to re-run this episode
                        waiting = False
                    self.clock.tick(self.target_fps * self.speed)  # Use speed multiplier here
                
        if self.visualize:
            pygame.quit()
            
        # Calculate averages
        avg_metrics = {
            metric: np.mean(values)
            for metric, values in all_metrics.items()
        }
        
        print("\nAverage metrics over episodes:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.2f}")
            
        return avg_metrics 