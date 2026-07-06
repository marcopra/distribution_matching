#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate dist_matching

export HYDRA_FULL_ERROR=1

DEVICE="${DEVICE:-cuda}"

python pretrain.py \
    --config-name=pretrain/pretrain_umaze_baselines \
    env=pointmaze/pointmaze_largedense_goal_1 \
    num_seed_frames=1000 \
    agent="${AGENT}" \
    seed="${SEED}" \
    device="${DEVICE}" \
    wandb_tag="${AGENT}"
