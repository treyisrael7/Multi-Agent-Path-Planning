"""Visualizer for the grid world environment"""

import pygame
import time
from visualization.renderer import Renderer
from visualization.controls import Controls

class Visualizer:
    """Visualizes the grid world environment"""
    
    def __init__(self, env, fps=10):
        pygame.init()
        self.env = env
        self.fps = fps
        self.clock = pygame.time.Clock()
        
        # Initialize components
        self.renderer = Renderer(env.size, env.config.cell_size)
        self.controls = Controls()
        
    def run(self):
        """Run the visualization"""
        while self.controls.should_continue():
            # Handle user input
            self.controls.handle_events()
            
            # Check for reset
            if self.controls.check_reset():
                self.env.reset()
                self.renderer.reset_stats()
            
            # Update if not paused
            if not self.controls.is_paused():
                # Get current state
                state = self.env.get_state()
                
                # Step environment
                grid, agent_positions, goal_positions, done, info = self.env.step()
                moves = info['moves']
                goals_collected = info['goals_collected']
                
                # Render current state
                self.renderer.render(state['grid'], state['paths'], moves, goals_collected)
                
                # Control frame rate and add small delay for smoother visualization
                self.clock.tick(self.fps)
                time.sleep(0.1)  # Add 100ms delay between steps
                
                if done:
                    time.sleep(1)  # Pause briefly when complete
                    self.env.reset()
                    self.renderer.reset_stats()
            
            else:
                # If paused, just render current state
                state = self.env.get_state()
                self.renderer.render(state['grid'], state['paths'])
                self.clock.tick(5)  # Even slower update when paused
        
        pygame.quit() 