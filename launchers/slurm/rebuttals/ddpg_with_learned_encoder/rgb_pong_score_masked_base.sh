#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=11:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate dist_matching


export HYDRA_FULL_ERROR=1


: "${REPLAY_BUFFER_DIR:?REPLAY_BUFFER_DIR is not set}"
: "${ENCODER_PATH:?ENCODER_PATH is not set}"


python train_offline.py agent=ddpg_discrete_with_learned_encoder replay_buffer_dir="${REPLAY_BUFFER_DIR}" env=pong_score_masked use_wandb=true agent.feature_dim=$FEATURE_DIM seed=$SEED num_grad_steps=110000 +encoder_path="${ENCODER_PATH}" grayscale=false