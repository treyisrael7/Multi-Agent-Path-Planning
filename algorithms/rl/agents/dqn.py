"""Deep Q-Network agent for pathfinding"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """Deep Q-Network model"""
    
    def __init__(self, state_shape, action_size):
        """Initialize the network
        
        Args:
            state_shape: Shape of state space (channels, height, width)
            action_size: Number of possible actions
        """
        super(DQN, self).__init__()
        
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
        conv_out_size = 64 * (state_shape[1] // 8) * (state_shape[2] // 8)  # state_shape[1] and [2] are height and width
        
        # Fully connected layers with dropout
        self.fc1 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(256, action_size)
        
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
            Q-values for each action
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class DQNAgent:
    """DQN agent for pathfinding"""
    
    def __init__(self, state_shape, action_size, num_agents=3, learning_rate=0.0001, 
                 memory_size=50000, batch_size=32, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update=500):
        """Initialize DQN agent
        
        Args:
            state_shape: Shape of state observations
            action_size: Number of possible actions per agent
            num_agents: Number of agents
            learning_rate: Learning rate for optimizer
            memory_size: Size of replay memory
            batch_size: Batch size for training
            gamma: Discount factor
            epsilon_start: Starting epsilon value
            epsilon_end: Minimum epsilon value
            epsilon_decay: Epsilon decay rate
            target_update: Number of steps between target network updates
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_agents = num_agents
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        
        # Initialize networks for each agent
        self.policy_nets = [DQN(state_shape, action_size).to(device) for _ in range(num_agents)]
        self.target_nets = [DQN(state_shape, action_size).to(device) for _ in range(num_agents)]
        for i in range(num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())
            self.target_nets[i].eval()  # Set target networks to eval mode
        
        # Initialize optimizers with gradient clipping
        self.optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in self.policy_nets]
        
        # Initialize replay memories
        self.memories = [deque(maxlen=memory_size) for _ in range(num_agents)]
        
        # Initialize metrics
        self.total_rewards = [0] * num_agents
        self.episode_rewards = [[] for _ in range(num_agents)]
        self.losses = [[] for _ in range(num_agents)]
        self.steps = 0
        
        # Pre-allocate tensors for batch processing
        # State shape is [C, H, W]
        self.states_batch = torch.zeros((batch_size, state_shape[0], state_shape[1], state_shape[2]), device=device)
        self.actions_batch = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.rewards_batch = torch.zeros((batch_size,), device=device)
        self.next_states_batch = torch.zeros((batch_size, state_shape[0], state_shape[1], state_shape[2]), device=device)
        self.dones_batch = torch.zeros((batch_size,), device=device)
        
    def select_action(self, state, evaluate=False):
        """Select actions for all agents using epsilon-greedy policy
        
        Args:
            state: Current state
            evaluate: Whether in evaluation mode
            
        Returns:
            List of selected actions and Q-value means
        """
        actions = []
        q_means = []
        
        # Convert state to tensor - it's already in [C, H, W] format
        # Add batch dimension and move to device
        state_tensor = torch.from_numpy(state).float().to(device)
        if len(state_tensor.shape) == 3:  # Add batch dimension if not present
            state_tensor = state_tensor.unsqueeze(0)
            
        for i in range(self.num_agents):
            if evaluate:
                with torch.no_grad():
                    q_values = self.policy_nets[i](state_tensor)
                    q_means.append(q_values.mean().item())  # Add mean Q-value
                    # Get action with highest Q-value
                    action = q_values.max(1)[1].item()
                    actions.append(action)
            else:
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    actions.append(random.randrange(self.action_size))
                    q_means.append(0.0)  # No Q-value for random actions
                else:
                    with torch.no_grad():
                        q_values = self.policy_nets[i](state_tensor)
                        q_means.append(q_values.mean().item())
                        action = q_values.max(1)[1].item()
                        actions.append(action)
                        
        return actions, q_means
            
    def store_transition(self, state, actions, rewards, next_state, done):
        """Store transitions in replay memory
        
        Args:
            state: Current state
            actions: List of actions taken by each agent
            rewards: List of rewards received by each agent
            next_state: Next state
            done: Whether episode is done
        """
        for i in range(self.num_agents):
            self.memories[i].append((state, actions[i], rewards[i], next_state, done))
        
    def update(self):
        """Update policy networks using experience replay"""
        if len(self.memories[0]) < self.batch_size:
            return None
            
        try:
            total_loss = 0
            
            # Update each agent's network
            for i in range(self.num_agents):
                # Sample batch from memory
                batch = random.sample(self.memories[i], self.batch_size)
                
                # Efficiently convert batch to tensors
                for j, (state, action, reward, next_state, done) in enumerate(batch):
                    # Convert states to tensors - they're already in [C, H, W] format
                    state_tensor = torch.from_numpy(state).float()
                    next_state_tensor = torch.from_numpy(next_state).float()
                    
                    # Move to device
                    self.states_batch[j] = state_tensor.to(device)
                    self.next_states_batch[j] = next_state_tensor.to(device)
                    self.actions_batch[j] = torch.tensor(action, dtype=torch.long, device=device)
                    self.rewards_batch[j] = torch.tensor(reward, dtype=torch.float, device=device)
                    self.dones_batch[j] = torch.tensor(done, dtype=torch.float, device=device)
                
                # Compute Q(s_t, a)
                current_q_values = self.policy_nets[i](self.states_batch).gather(1, self.actions_batch.unsqueeze(1))
                
                # Compute V(s_{t+1})
                with torch.no_grad():
                    next_q_values = self.target_nets[i](self.next_states_batch).max(1)[0]
                    expected_q_values = self.rewards_batch + (1 - self.dones_batch) * self.gamma * next_q_values
                    
                # Compute Huber loss
                loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
                
                # Optimize
                self.optimizers[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 0.5)
                self.optimizers[i].step()
                
                # Store loss
                self.losses[i].append(loss.item())
                total_loss += loss.item()
                
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target networks
            self.steps += 1
            if self.steps % self.target_update == 0:
                for i in range(self.num_agents):
                    self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())
                    
            return total_loss / self.num_agents
            
        except Exception as e:
            print(f"Error in DQN update: {e}")
            return None
        
    def save(self, path):
        """Save agent to file
        
        Args:
            path: Path to save to
        """
        # Move networks to CPU before saving
        for i in range(self.num_agents):
            self.policy_nets[i] = self.policy_nets[i].cpu()
            self.target_nets[i] = self.target_nets[i].cpu()
        
        torch.save({
            'policy_nets': [net.state_dict() for net in self.policy_nets],
            'target_nets': [net.state_dict() for net in self.target_nets],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'epsilon': self.epsilon,
            'memories': [list(mem) for mem in self.memories],
            'episode_rewards': [self.episode_rewards[i] for i in range(self.num_agents)],
            'losses': [self.losses[i] for i in range(self.num_agents)]
        }, path)
        
        # Move networks back to GPU
        for i in range(self.num_agents):
            self.policy_nets[i] = self.policy_nets[i].to(device)
            self.target_nets[i] = self.target_nets[i].to(device)
        torch.cuda.empty_cache()
        
    def load(self, path):
        """Load agent from file
        
        Args:
            path: Path to load from
        """
        checkpoint = torch.load(path)
        for i in range(self.num_agents):
            self.policy_nets[i].load_state_dict(checkpoint['policy_nets'][i])
            self.target_nets[i].load_state_dict(checkpoint['target_nets'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizers'][i])
        self.epsilon = checkpoint['epsilon']
        for i in range(self.num_agents):
            self.memories[i] = deque(checkpoint['memories'][i], maxlen=self.memory_size)
            self.episode_rewards[i] = checkpoint['episode_rewards'][i]
            self.losses[i] = checkpoint['losses'][i]
        torch.cuda.empty_cache() 