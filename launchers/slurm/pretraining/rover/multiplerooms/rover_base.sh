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


python pretrain.py --config-name=pretrain/pretrain_rover_multiplerooms use_wandb=true env=${ENV} device=cuda seed=${SEED} save_video=false save_buffer=false +save_snapshot=false agent.batch_size_actor=${BATCH_SIZE_ACTOR} agent.feature_dim=200