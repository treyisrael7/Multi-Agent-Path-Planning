"""Advantage Actor-Critic (A2C) implementation for pathfinding"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
import os
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    """Actor-Critic network for A2C"""
    
    def __init__(self, state_shape, action_size):
        """Initialize the network
        
        Args:
            state_shape: Shape of state space (channels, height, width)
            action_size: Number of possible actions
        """
        super(ActorCritic, self).__init__()
        
        # Convolutional layers with batch normalization and max pooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=3, padding=1),  # state_shape[0] is channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduces spatial size by half
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduces spatial size by half again
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Final reduction
        )
        
        # Calculate size of flattened features after max pooling
        # Each max pool reduces spatial size by half (3 max pools = 1/8 size)
        conv_out_size = 64 * (state_shape[1] // 8) * (state_shape[2] // 8)
        
        # Shared feature layers with dropout
        self.features = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared features
        features = self.features(x)
        
        # Actor and critic heads
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value

class A2CAgent:
    """A2C agent for pathfinding"""
    
    def __init__(self, state_shape, action_size, num_agents=3, learning_rate=0.001, 
                 gamma=0.99, update_frequency=32):
        """Initialize A2C agent
        
        Args:
            state_shape: Shape of state observations
            action_size: Number of possible actions per agent
            num_agents: Number of agents
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            update_frequency: Number of steps between updates
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.update_frequency = update_frequency
        
        # Initialize networks for each agent
        self.actor_critics = [ActorCritic(state_shape, action_size).to(device) for _ in range(num_agents)]
        
        # Initialize optimizers with gradient clipping
        self.optimizers = [optim.Adam(ac.parameters(), lr=learning_rate) for ac in self.actor_critics]
        
        # Initialize metrics
        self.total_rewards = [0] * num_agents
        self.episode_rewards = [[] for _ in range(num_agents)]
        self.actor_losses = [[] for _ in range(num_agents)]
        self.critic_losses = [[] for _ in range(num_agents)]
        
        # Initialize memory for episode data
        self.reset_memory()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
    def reset_memory(self):
        """Reset episode memory"""
        self.rewards = [[] for _ in range(self.num_agents)]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.state_values = [[] for _ in range(self.num_agents)]
        self.steps_since_update = 0
        
    def select_action(self, state):
        """Select actions for all agents using policy networks
        
        Args:
            state: Current state
            
        Returns:
            List of selected actions
        """
        actions = []
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        
        for i in range(self.num_agents):
            # Get action probabilities and state value
            action_probs, state_value = self.actor_critics[i](state)
            
            # Sample action from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            # Store log probability and state value
            self.log_probs[i].append(action_dist.log_prob(action))
            self.state_values[i].append(state_value)
            
            actions.append(action.item())
            
        return actions
            
    def store_reward(self, rewards):
        """Store rewards for current step
        
        Args:
            rewards: List of rewards for each agent
        """
        for i, reward in enumerate(rewards):
            self.rewards[i].append(reward)
        self.steps_since_update += 1
        
        # Update if enough steps collected
        if self.steps_since_update >= self.update_frequency:
            self.update()
            self.steps_since_update = 0
        
    def update(self):
        """Update networks using collected experience"""
        if len(self.rewards[0]) < 2:  # Need at least 2 steps for advantage calculation
            return None, None
            
        try:
            total_actor_loss = 0
            total_critic_loss = 0
            
            # Update each agent's network
            for i in range(self.num_agents):
                # Convert lists to tensors and move to device
                rewards = torch.from_numpy(np.array(self.rewards[i])).to(device)
                log_probs = torch.stack(self.log_probs[i]).to(device)
                state_values = torch.stack(self.state_values[i]).to(device)
                
                # Calculate returns and advantages
                returns = []
                advantages = []
                R = 0
                
                for r, v in zip(reversed(self.rewards[i]), reversed(state_values)):
                    R = r + self.gamma * R
                    advantage = R - v.item()
                    returns.insert(0, R)
                    advantages.insert(0, advantage)
                    
                # Convert to tensors and enable gradients
                returns = torch.from_numpy(np.array(returns)).to(device)
                returns.requires_grad_(True)
                advantages = torch.from_numpy(np.array(advantages)).to(device)
                advantages.requires_grad_(True)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Calculate losses with entropy bonus
                actor_loss = -(log_probs * advantages.detach()).mean()
                critic_loss = F.smooth_l1_loss(state_values.squeeze(), returns.detach())
                
                # Add entropy bonus to encourage exploration
                entropy = -(log_probs * torch.exp(log_probs)).mean()
                entropy_bonus = 0.01 * entropy
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss - entropy_bonus
                
                # Optimize
                self.optimizers[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critics[i].parameters(), 0.5)
                self.optimizers[i].step()
                
                # Store losses
                self.actor_losses[i].append(actor_loss.item())
                self.critic_losses[i].append(critic_loss.item())
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                
                # Clear memory after update
                self.rewards[i] = []
                self.log_probs[i] = []
                self.state_values[i] = []
                
            return total_actor_loss / self.num_agents, total_critic_loss / self.num_agents
            
        except Exception as e:
            print(f"Error in A2C update: {e}")
            return None, None
            
    def save(self, path):
        """Save agent to file
        
        Args:
            path: Path to save to
        """
        torch.save({
            'actor_critics': [ac.state_dict() for ac in self.actor_critics],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'episode_rewards': [ep for ep in self.episode_rewards],
            'actor_losses': [loss for loss in self.actor_losses],
            'critic_losses': [loss for loss in self.critic_losses]
        }, path)
        
    def load(self, path):
        """Load agent from file
        
        Args:
            path: Path to load from
        """
        checkpoint = torch.load(path)
        for i, ac in enumerate(self.actor_critics):
            ac.load_state_dict(checkpoint['actor_critics'][i])
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(checkpoint['optimizers'][i])
        for i, ep in enumerate(self.episode_rewards):
            ep.extend(checkpoint['episode_rewards'][i])
        for i, loss in enumerate(self.actor_losses):
            loss.extend(checkpoint['actor_losses'][i])
        for i, loss in enumerate(self.critic_losses):
            loss.extend(checkpoint['critic_losses'][i]) 