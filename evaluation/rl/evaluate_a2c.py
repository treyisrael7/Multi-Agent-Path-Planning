"""Script to evaluate a trained A2C agent"""

import argparse
from evaluation.rl.a2c_evaluator import A2CEvaluator

def main():
    parser = argparse.ArgumentParser(description='Evaluate A2C agent')
    parser.add_argument('--model_path', type=str, default='models/a2c/a2c_model.pth',
                      help='Path to the trained model checkpoint')
    parser.add_argument('--no-vis', action='store_true',
                      help='Disable visualization')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    
    args = parser.parse_args()
    
    evaluator = A2CEvaluator(
        model_path=args.model_path,
        visualize=not args.no_vis
    )
    
    evaluator.evaluate(num_episodes=args.episodes)

if __name__ == '__main__':
    main() 