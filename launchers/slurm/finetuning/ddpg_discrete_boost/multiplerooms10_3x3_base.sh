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


python train.py agent=ddpg_discrete_with_kernel_actor eval_every_frames=5000 num_train_frames=100000 configs/env=${ENV} device=cuda seed=${SEED} env.render_mode=null save_video=false wandb_tag=boost p_path=${MODEL_PATH} agent.actor_lr=1e-7 use_wandb=true save_train_video=false agent.feature_dim=150 agent.dataset_dim=10000 num_seed_frames=2000

