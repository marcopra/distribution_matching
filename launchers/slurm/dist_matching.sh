#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=cpu

cd $SLURM_SUBMIT_DIR


export HYDRA_FULL_ERROR=1


# Load environment
source ~/.bashrc
conda activate dist_matching

# Use quotes and escape model path appropriately
python3 train_metaworld.py agent.pretrained_path=\"${MODEL_PATH}\" exp_name=\"${EXP_NAME}\" seed=$SEED_ARG env_name=$ENV_NAME random_init=$RANDOM_HAND random_goal=$RANDOM_GOAL wandb_tag=$WANDB_TAG wandb_project=$WANDB_PROJECT num_train_frames=220000 agent.freeze_encoder=$FREEZE agent.no_taco=$NO_TACO agent=$AGENT
