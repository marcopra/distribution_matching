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

# Use environment variables passed via sbatch
SEED=${SEED:-0}
SEED_FRAMES=${SEED_FRAMES:-4000}
MODEL_PATH=${MODEL_PATH:-""}
CFG_ENV=${CFG_ENV:-"four_rooms10_2"}



# Load environment
source ~/.bashrc
conda activate dist_matching


export HYDRA_FULL_ERROR=1


python train.py agent=sac_discrete use_wandb=true num_seed_frames=${SEED_FRAMES} eval_every_frames=20000 num_train_frames=200000 p_path="${MODEL_PATH}" configs/env=${CFG_ENV} device=cuda seed=${SEED} save_video=false env.render_mode=null wandb_tag="sac_discrete"
