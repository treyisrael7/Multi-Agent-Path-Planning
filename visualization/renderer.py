"""Renderer for the grid world visualization"""

import pygame
import time

class Renderer:
    """Handles rendering of the grid world"""
    
    def __init__(self, grid_size, cell_size):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_pixels = grid_size * cell_size
        self.panel_width = 300  # Width of the status panel
        self.screen = pygame.display.set_mode((self.grid_pixels + self.panel_width, self.grid_pixels))
        pygame.display.set_caption("Grid World Pathfinding")
        
        # Initialize fonts
        self.font = pygame.font.Font(None, cell_size)  # For grid numbers
        self.status_font = pygame.font.Font(None, 36)  # For status panel
        
        # Colors
        self.COLORS = {
            'background': (255, 255, 255),  # White
            'grid': (200, 200, 200),        # Light gray
            'obstacle': (100, 100, 100),    # Dark gray
            'goal': (0, 255, 0),            # Green
            'agent': (255, 100, 0),         # Orange
            'path': (0, 0, 255, 50),        # Semi-transparent blue
            'panel': (240, 240, 240),       # Light gray for panel
            'text': (0, 0, 0)               # Black text
        }
        
        # Stats
        self.stats = {
            'steps': 0,
            'goals_collected': 0,
            'start_time': time.time()
        }
        
    def draw_grid(self):
        """Draw the grid lines"""
        for i in range(self.grid_size):
            # Vertical lines
            pygame.draw.line(
                self.screen, 
                self.COLORS['grid'],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.grid_pixels)
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                self.COLORS['grid'],
                (0, i * self.cell_size),
                (self.grid_pixels, i * self.cell_size)
            )
            
    def draw_cell(self, x, y, color, number=None):
        """Draw a colored cell with optional number"""
        pygame.draw.rect(
            self.screen,
            color,
            (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
        )
        
        if number is not None:
            text = self.font.render(str(number), True, (255, 255, 255) if color == self.COLORS['agent'] else (0, 0, 0))
            text_rect = text.get_rect(center=(
                y * self.cell_size + self.cell_size // 2,
                x * self.cell_size + self.cell_size // 2
            ))
            self.screen.blit(text, text_rect)
            
    def draw_path(self, path):
        """Draw a path"""
        if len(path) < 2:
            return
            
        points = [(y * self.cell_size + self.cell_size // 2, 
                  x * self.cell_size + self.cell_size // 2) 
                 for x, y in path]
                 
        # Create a surface for the path
        path_surface = pygame.Surface((self.grid_pixels, self.grid_pixels), pygame.SRCALPHA)
        pygame.draw.lines(path_surface, self.COLORS['path'], False, points, 2)
        self.screen.blit(path_surface, (0, 0))
        
    def draw_status_panel(self, num_goals_remaining):
        """Draw the status panel"""
        panel_rect = pygame.Rect(self.grid_pixels, 0, self.panel_width, self.grid_pixels)
        pygame.draw.rect(self.screen, self.COLORS['panel'], panel_rect)
        
        # Update stats
        elapsed_time = int(time.time() - self.stats['start_time'])
        
        # Status text
        texts = [
            f"Steps: {self.stats['steps']}",
            f"Goals Collected: {self.stats['goals_collected']}",
            f"Goals Remaining: {num_goals_remaining}",
            f"Time: {elapsed_time}s",
            "",
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset",
            "ESC - Quit"
        ]
        
        for i, text in enumerate(texts):
            surface = self.status_font.render(text, True, self.COLORS['text'])
            self.screen.blit(surface, (self.grid_pixels + 20, 20 + i * 40))
        
    def render(self, grid, paths=None, moves=None, goals_collected=0):
        """Render the current state"""
        self.screen.fill(self.COLORS['background'])
        
        # Update stats
        if moves:
            self.stats['steps'] += 1
        if goals_collected > 0:
            self.stats['goals_collected'] += goals_collected
        
        # Draw grid contents
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cell = grid[i, j]
                if cell == 1:  # Obstacle
                    self.draw_cell(i, j, self.COLORS['obstacle'])
                elif cell == 2:  # Goal
                    self.draw_cell(i, j, self.COLORS['goal'])
                elif cell == 3:  # Agent
                    agent_idx = next((idx for idx, pos in enumerate(paths) if pos and pos[0] == (i, j)), None)
                    self.draw_cell(i, j, self.COLORS['agent'], agent_idx + 1 if agent_idx is not None else None)
        
        # Draw paths
        if paths:
            for path in paths:
                if path:
                    self.draw_path(path)
        
        # Draw grid lines
        self.draw_grid()
        
        # Draw status panel
        num_goals = sum(1 for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == 2)
        self.draw_status_panel(num_goals)
        
        pygame.display.flip()
        
    def reset_stats(self):
        """Reset the statistics"""
        self.stats = {
            'steps': 0,
            'goals_collected': 0,
            'start_time': time.time()
        } 