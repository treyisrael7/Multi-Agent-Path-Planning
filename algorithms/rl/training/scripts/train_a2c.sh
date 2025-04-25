#!/bin/bash
#SBATCH --job-name=a2c_train_new
#SBATCH --output=a2c_training_new.out
#SBATCH --error=a2c_training_new.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --partition=work1
#SBATCH --cpus-per-task=8
#SBATCH --qos=normal

# Print start time and node information
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"

# Load modules
module load miniforge3

# Activate conda environment
source activate path_planning

# Set environment variables for memory management
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Print environment information
echo "Python path: $(which python)"
echo "Using device: cpu"
echo "CPU cores: $(nproc)"

# Run A2C training
python -u -m algorithms.rl.train --agent a2c --episodes 1000 --grid_size 50 --num_agents 3 --num_goals 50 --num_obstacles 15 