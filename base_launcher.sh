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

# python pretrain.py agent=dist_matching_embedding_augmented save_video=false num_train_frames=300000 use_wandb=false agent.T_init_steps=20000 configs/env=multiplerooms10_3x3_0 agent.data_type=all num_seed_frames=2000 agent.update_actor_every_steps=1500 agent.window_size=10 agent.unique_window=false agent.n_subsamples=5000 agent.subsampling_strategy=eder agent.epsilon_schedule=0.15 "agent.sink_schedule='linear(0.0,0.004,400000)'" agent.T_init_steps=50 agent.feature_dim=200 agent.lr_actor=10 agent.pmd_steps=250 obs_type=pixels agent.lambda_reg=1e-6
python pretrain.py agent=dist_matching_embedding_augmented save_video=false num_train_frames=300000 use_wandb=false agent.T_init_steps=20000 configs/env=two_roomstwo_rooms7_0 agent.data_type=all num_seed_frames=2000 agent.update_actor_every_steps=1500 agent.window_size=10 agent.unique_window=false agent.n_subsamples=5000 agent.subsampling_strategy=eder agent.epsilon_schedule=0.15 "agent.sink_schedule='linear(0.0,0.004,40000)'" agent.T_init_steps=50 agent.feature_dim=200 agent.lr_actor=10 agent.pmd_steps=250 obs_type=pixels agent.lambda_reg=1e-6
