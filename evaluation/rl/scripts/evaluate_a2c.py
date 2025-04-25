import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from algorithms.rl.environment.pathfinding_env import PathfindingEnv
from algorithms.rl.agents.a2c import A2CAgent

def load_model(model_path, grid_size=50, num_agents=3):
    """Load A2C model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        grid_size: Size of the grid
        num_agents: Number of agents
        
    Returns:
        Loaded agent and environment
    """
    # Create environment
    env = PathfindingEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_goals=50,
        num_obstacles=15
    )
    
    # Create agent
    agent = A2CAgent(
        state_shape=(grid_size, grid_size, 3),
        action_size=8
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    for i in range(num_agents):
        agent.actor_critics[i].load_state_dict(checkpoint['actor_critics'][i])
    
    return agent, env

def evaluate_model(agent, env, num_episodes=10):
    """Evaluate the model
    
    Args:
        agent: A2C agent
        env: Environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    eval_rewards = []
    eval_steps = []
    eval_goals = []
    eval_path_lengths = []
    
    for episode in range(num_episodes):
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
        eval_path_lengths.append(info['path_length'])
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Average Reward: {eval_rewards[-1]:.2f}")
        print(f"  Steps: {eval_steps[-1]}")
        print(f"  Goals Collected: {eval_goals[-1]}")
        print(f"  Path Length: {eval_path_lengths[-1]}")
        
    return {
        'reward': np.mean(eval_rewards),
        'steps': np.mean(eval_steps),
        'goals': np.mean(eval_goals),
        'path_length': np.mean(eval_path_lengths),
        'reward_std': np.std(eval_rewards),
        'steps_std': np.std(eval_steps),
        'goals_std': np.std(eval_goals),
        'path_length_std': np.std(eval_path_lengths)
    }

def plot_evaluation_results(results):
    """Plot evaluation results
    
    Args:
        results: Dictionary of evaluation metrics
    """
    metrics = ['reward', 'steps', 'goals', 'path_length']
    values = [results[m] for m in metrics]
    stds = [results[f'{m}_std'] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, yerr=stds, capsize=5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    ax.set_title('A2C Model Evaluation Results')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('a2c_evaluation_results.png')
    plt.close()

def main():
    # Load model
    model_path = Path("models/a2c/checkpoint_episode_800.pth")
    agent, env = load_model(model_path)
    
    # Evaluate model
    print("\nEvaluating A2C model...")
    results = evaluate_model(agent, env, num_episodes=10)
    
    # Print results
    print("\nFinal Evaluation Results:")
    print(f"Average Reward: {results['reward']:.2f} ± {results['reward_std']:.2f}")
    print(f"Average Steps: {results['steps']:.2f} ± {results['steps_std']:.2f}")
    print(f"Average Goals: {results['goals']:.2f} ± {results['goals_std']:.2f}")
    print(f"Average Path Length: {results['path_length']:.2f} ± {results['path_length_std']:.2f}")
    
    # Plot results
    plot_evaluation_results(results)
    print("\nResults plot saved as 'a2c_evaluation_results.png'")

if __name__ == "__main__":
    main() 