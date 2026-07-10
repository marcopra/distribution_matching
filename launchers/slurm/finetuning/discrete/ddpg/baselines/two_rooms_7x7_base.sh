#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate dist_matching


export HYDRA_FULL_ERROR=1


python train.py agent=ddpg_discrete eval_every_frames=5000 num_train_frames=200000 configs/env=${ENV} device=cuda seed=${SEED} save_video=false p_path=${MODEL_PATH} use_wandb=true save_train_video=false  agent.feature_dim=200 agent.actor_lr=1e-7 agent.init_critic=false num_seed_frames=2000 obs_type=${OBS_TYPE} env.render_mode=null wandb_project=finetune_debug

