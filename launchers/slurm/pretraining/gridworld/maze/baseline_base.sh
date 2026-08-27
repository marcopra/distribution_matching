#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate dist_matching

export HYDRA_FULL_ERROR=1

python pretrain.py \
    --config-name=pretrain/pretrain_maze_scaling_baselines \
    agent="${AGENT}" \
    env="${ENV}" \
    seed="${SEED}"

