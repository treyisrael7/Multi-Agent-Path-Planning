# Path Planning with Reinforcement Learning

This project implements a multi-agent path planning system using reinforcement learning, specifically PPO (Proximal Policy Optimization).

## Project Structure
```
.
├── env/                    # Environment implementation
│   └── multi_agent_env.py # Grid world environment
├── models/                 # Model implementations and training
│   ├── train_ppo.py       # PPO training script
│   ├── evaluate_model.py  # Model evaluation
│   └── common.py          # Common utilities for models
├── utils/                 # Utility functions
│   └── visualize.py      # Visualization tools
└── scripts/              # Training and evaluation scripts
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:
```bash
python models/train_ppo.py
```

## Evaluation

To evaluate a trained model:
```bash
python models/evaluate_model.py
```

## Environment

The environment is a grid world where agents need to reach goals while avoiding obstacles:
- Multiple agents can be present
- Multiple goals to collect
- Obstacles to avoid
- Continuous rewards for progress towards goals
- Early termination on collision with obstacles

## Model Architecture

Using PPO with:
- MLP Policy network: [128, 128, 64]
- Value network: [128, 128, 64]
- Learning rate: 1e-4
- Parallel environments: 8
- Batch size: 128
