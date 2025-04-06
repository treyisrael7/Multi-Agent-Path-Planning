"""Controls handler for the grid world visualization"""

import pygame

class Controls:
    """Handles user input and simulation controls"""
    
    def __init__(self):
        self.paused = False
        self.running = True
        self.reset_requested = False
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_requested = True
                    
    def should_continue(self):
        """Check if the simulation should continue running"""
        return self.running
        
    def is_paused(self):
        """Check if the simulation is paused"""
        return self.paused
        
    def check_reset(self):
        """Check and clear the reset flag"""
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False 