import numpy as np
import torch
from algorithms.ppo import PPOAgent
from env.grid_world import GridWorld
from visualization.visualizer import Visualizer
import time

def get_state(grid_world, agent_idx):
    """Convert grid world state to agent observation"""
    agent_pos = grid_world.agent_positions[agent_idx]
    
    # Get local view around agent (5x5 grid)
    view_size = 5
    half_size = view_size // 2
    view = np.zeros((view_size, view_size, 4))  # 4 channels: empty, obstacle, goal, agent
    
    for i in range(view_size):
        for j in range(view_size):
            x = agent_pos[0] - half_size + i
            y = agent_pos[1] - half_size + j
            
            if 0 <= x < grid_world.size and 0 <= y < grid_world.size:
                cell = grid_world.grid[x, y]
                view[i, j, max(0, cell-1)] = 1
    
    # Find closest goal
    closest_goal_dist = float('inf')
    closest_goal_dir = [0, 0]
    
    for goal in grid_world.goal_positions:
        dx = goal[0] - agent_pos[0]
        dy = goal[1] - agent_pos[1]
        dist = abs(dx) + abs(dy)
        if dist < closest_goal_dist:
            closest_goal_dist = dist
            closest_goal_dir = [dx / grid_world.size, dy / grid_world.size]
    
    # Flatten and concatenate
    state = np.concatenate([
        view.flatten(),
        closest_goal_dir,
        [closest_goal_dist / grid_world.size]
    ])
    
    return state

def train():
    # Environment setup
    grid_world = GridWorld()
    vis = Visualizer(grid_world)
    
    # Agent setup
    n_agents = grid_world.num_agents
    input_dims = 5*5*4 + 3  # 5x5 view (4 channels) + goal direction (2) + goal distance (1)
    n_actions = 4  # up, down, left, right
    agents = [PPOAgent(input_dims, n_actions) for _ in range(n_agents)]
    
    # Training parameters
    n_episodes = 1000
    max_steps = 200
    update_interval = 20
    
    # Training loop
    for episode in range(n_episodes):
        grid_world.reset()
        episode_rewards = [0 for _ in range(n_agents)]
        step_count = 0
        
        while step_count < max_steps:
            # Get states
            states = [get_state(grid_world, i) for i in range(n_agents)]
            
            # Get actions
            actions = []
            log_probs = []
            values = []
            
            for i, agent in enumerate(agents):
                action, log_prob, value = agent.choose_action(states[i])
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
            
            # Convert actions to moves
            moves = []
            for i, action in enumerate(actions):
                agent_pos = grid_world.agent_positions[i]
                if action == 0:  # up
                    new_pos = (max(0, agent_pos[0]-1), agent_pos[1])
                elif action == 1:  # down
                    new_pos = (min(grid_world.size-1, agent_pos[0]+1), agent_pos[1])
                elif action == 2:  # left
                    new_pos = (agent_pos[0], max(0, agent_pos[1]-1))
                else:  # right
                    new_pos = (agent_pos[0], min(grid_world.size-1, agent_pos[1]+1))
                moves.append((i, new_pos))
            
            # Execute moves
            done, _, goals_collected = grid_world.step()
            
            # Calculate rewards
            rewards = [0 for _ in range(n_agents)]
            for i in range(n_agents):
                if goals_collected > 0:
                    rewards[i] = 10.0  # Reward for collecting goals
                else:
                    # Small penalty for distance to closest goal
                    closest_dist = min((abs(grid_world.agent_positions[i][0] - g[0]) + 
                                      abs(grid_world.agent_positions[i][1] - g[1]))
                                     for g in grid_world.goal_positions) if grid_world.goal_positions else 0
                    rewards[i] = -0.1 * (closest_dist / grid_world.size)
            
            # Store transitions
            for i in range(n_agents):
                agents[i].store_transition(states[i], actions[i], log_probs[i], 
                                        values[i], rewards[i], done)
                episode_rewards[i] += rewards[i]
            
            # Update visualization
            vis.draw()
            
            if done:
                break
                
            step_count += 1
        
        # Learn
        if episode % update_interval == 0:
            for agent in agents:
                agent.learn()
        
        # Print progress
        avg_reward = sum(episode_rewards) / n_agents
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Steps: {step_count}")
        
        time.sleep(0.01)  # Small delay to prevent GPU overload

if __name__ == "__main__":
    train() 