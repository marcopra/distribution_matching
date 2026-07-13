#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd "$SLURM_SUBMIT_DIR" || exit 1

# Load environment
source ~/.bashrc
conda activate dist_matching

export HYDRA_FULL_ERROR=1

python pretrain_parallel.py \
    --config-name=pretrain_parallel/pretrain_montezouma_random \
    agent=random \
    wandb_tag=parallel_random \
    use_wandb=true \
    seed="${SEED}"
