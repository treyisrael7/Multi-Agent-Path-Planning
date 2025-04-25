"""Training script for RL agents in pathfinding environment"""

import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

from algorithms.rl.environment.pathfinding_env import PathfindingEnv
from algorithms.rl.agents.dqn import DQNAgent
from algorithms.rl.agents.a2c import A2CAgent

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def setup_logging(save_path):
    """Setup logging to file and console"""
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    log_path = os.path.join(save_path, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_agent(env, agent, num_episodes, save_dir, eval_interval=100):
    """Train an agent
    
    Args:
        env: Environment
        agent: Agent to train
        num_episodes: Number of episodes to train for
        save_dir: Directory to save checkpoints
        eval_interval: Interval for evaluation and checkpointing
    """
    # Force immediate output
    import sys
    sys.stdout.flush()
    
    print(f"Starting training with {num_episodes} episodes", flush=True)
    print(f"Environment: {env.grid_size}x{env.grid_size} grid, {env.num_agents} agents, {env.num_goals} goals, {env.num_obstacles} obstacles", flush=True)
    print(f"Saving checkpoints to: {save_dir}", flush=True)
    print(f"Using device: {device}", flush=True)
    
    best_reward = float('-inf')
    os.makedirs(save_dir, exist_ok=True)
    
    # Memory optimizations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    
    # Update frequency for DQN
    update_frequency = 10  # Update every 10 steps
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_rewards = [0] * env.num_agents
        steps = 0
        done = False
        episode_q_means = []
        episode_goals = 0
        
        # Reset agent memory at start of episode
        if hasattr(agent, 'reset_memory'):
            agent.reset_memory()
        
        while not done:
            # Get actions (and Q-values for DQN)
            if isinstance(agent, DQNAgent):
                actions, q_means = agent.select_action(state)
                episode_q_means.extend(q_means)
            else:  # A2C
                actions = agent.select_action(state)
                episode_q_means = [0.0]  # Placeholder for A2C
            
            # Take step
            next_state, rewards, done, _ = env.step(actions)
            
            # Store experience based on agent type
            if isinstance(agent, DQNAgent):
                agent.store_transition(state, actions, rewards, next_state, done)
            elif isinstance(agent, A2CAgent):
                agent.store_reward(rewards)
            
            # Update agent
            if isinstance(agent, DQNAgent):
                if len(agent.memories[0]) >= agent.batch_size and steps % update_frequency == 0:
                    loss = agent.update()
            elif isinstance(agent, A2CAgent):
                if done:  # A2C updates at the end of episode
                    agent.update()
            
            # Update metrics
            for i in range(env.num_agents):
                total_rewards[i] += rewards[i]
            steps += 1
            episode_goals += sum(1 for r in rewards if r == env.goal_reward)  # Count goals collected
            
            state = next_state
            
            # Clean up memory every 25 steps
            if steps % 25 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # Print progress every episode
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_q_mean = sum(episode_q_means) / len(episode_q_means) if episode_q_means else 0
        avg_steps_per_goal = steps / (episode_goals or 1)  # Avoid division by zero
        print(f"Episode {episode}/{num_episodes}, Steps: {steps}, Average Reward: {avg_reward:.2f}, "
              f"Average Q: {avg_q_mean:.2f}, Goals: {episode_goals}, "
              f"Steps per Goal: {avg_steps_per_goal:.2f}", flush=True)
        
        # Save checkpoint and cleanup every eval_interval episodes
        if episode % eval_interval == 0:
            # Force cleanup after logging
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                checkpoint_path = os.path.join(save_dir, f'checkpoint_episode_{episode}.pth')
                
                # Save based on agent type
                if isinstance(agent, DQNAgent):
                    # Move models to CPU before saving
                    for i in range(agent.num_agents):
                        agent.policy_nets[i] = agent.policy_nets[i].cpu()
                        agent.target_nets[i] = agent.target_nets[i].cpu()
                    torch.save({
                        'episode': episode,
                        'policy_nets': [net.state_dict() for net in agent.policy_nets],
                        'target_nets': [net.state_dict() for net in agent.target_nets],
                        'optimizers': [opt.state_dict() for opt in agent.optimizers],
                        'reward': avg_reward,
                    }, checkpoint_path)
                    # Move models back to GPU
                    for i in range(agent.num_agents):
                        agent.policy_nets[i] = agent.policy_nets[i].to(device)
                        agent.target_nets[i] = agent.target_nets[i].to(device)
                elif isinstance(agent, A2CAgent):
                    # Move models to CPU before saving
                    for i in range(agent.num_agents):
                        agent.actor_critics[i] = agent.actor_critics[i].cpu()
                    torch.save({
                        'episode': episode,
                        'actor_critics': [ac.state_dict() for ac in agent.actor_critics],
                        'optimizers': [opt.state_dict() for opt in agent.optimizers],
                        'reward': avg_reward,
                    }, checkpoint_path)
                    # Move models back to GPU
                    for i in range(agent.num_agents):
                        agent.actor_critics[i] = agent.actor_critics[i].to(device)
                
                print(f"Saved checkpoint at episode {episode} with reward {avg_reward:.2f}", flush=True)
                # Clean up after saving
                torch.cuda.empty_cache()
                gc.collect()
    
    return agent

def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate the agent
    
    Args:
        env: Gym environment
        agent: RL agent
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    eval_rewards = []
    eval_steps = []
    eval_goals = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = [0] * env.num_agents
        episode_steps = 0
        
        while not done:
            # Select actions in evaluation mode
            actions = agent.select_action(state, evaluate=True)
            
            # Execute actions
            next_state, rewards, done, info = env.step(actions)
            
            # Update metrics
            for i in range(env.num_agents):
                episode_rewards[i] += rewards[i]
            episode_steps += 1
            state = next_state
            
        eval_rewards.append(sum(episode_rewards) / len(episode_rewards))
        eval_steps.append(episode_steps)
        eval_goals.append(info['goals_collected'])
        
    return {
        'reward': np.mean(eval_rewards),
        'steps': np.mean(eval_steps),
        'goals': np.mean(eval_goals)
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train RL agents for pathfinding')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'a2c', 'both'],
                      help='Agent type to train')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of training episodes')
    parser.add_argument('--grid_size', type=int, default=50,
                      help='Size of the grid')
    parser.add_argument('--num_agents', type=int, default=3,
                      help='Number of agents')
    parser.add_argument('--num_goals', type=int, default=50,
                      help='Number of goals')
    parser.add_argument('--num_obstacles', type=int, default=15,
                      help='Number of obstacles')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from checkpoint')
    args = parser.parse_args()
    
    # Create environment
    env = PathfindingEnv(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        num_goals=args.num_goals,
        num_obstacles=args.num_obstacles
    )
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'runs/{args.agent}_{timestamp}')
    
    # Train DQN
    if args.agent in ['dqn', 'both']:
        dqn_agent = DQNAgent(
            state_shape=(3, args.grid_size, args.grid_size),  # [C, H, W] format
            action_size=8  # 8 possible movements
        )
        train_agent(env, dqn_agent, args.episodes, save_dir / 'dqn', eval_interval=100)
        
    # Train A2C
    if args.agent in ['a2c', 'both']:
        a2c_agent = A2CAgent(
            state_shape=(args.grid_size, args.grid_size, 3),
            action_size=8  # 8 possible movements
        )
        train_agent(env, a2c_agent, args.episodes, save_dir / 'a2c', eval_interval=100)

if __name__ == '__main__':
    main() 