#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate cleanrl311

export HYDRA_FULL_ERROR=1




python cleanrl_rnd_ppo.py \
  --track \
  --wandb_project_name montezuma_hp \
  --
  --seed 1
