#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv-longrun

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate dist_matching

export HYDRA_FULL_ERROR=1

DEVICE="${DEVICE:-cuda}"

python pretrain.py \
    --config-name=pretrain/pretrain_umaze_baselines \
    agent="${AGENT}" \
    seed="${SEED}" \
    device="${DEVICE}" \
    wandb_tag="${AGENT}"
