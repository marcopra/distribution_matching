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
conda activate dist_matching


export HYDRA_FULL_ERROR=1


python pretrain.py agent=smm_discrete use_wandb=true eval_every_frames=20000 num_train_frames=500000 agent.feature_dim=200 configs/env=${ENV} device=cuda seed=${SEED} save_video=false env.render_mode=null wandb_tag="smm_discrete"
