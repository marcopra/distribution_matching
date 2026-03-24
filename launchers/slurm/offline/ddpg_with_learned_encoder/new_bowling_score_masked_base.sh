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


python train_offline.py agent=ddpg_discrete_with_learned_encoder replay_buffer_dir="${REPLAY_BUFFER_DIR}" env=bowling_score_masked use_wandb=true seed=$SEED num_grad_steps=500000 +encoder_path=/home/mprattico/distribution_matching/data_offline/bowling_score_masked/1M/rover_50/models/pixels/gym/dist_matching/1/snapshot.pt