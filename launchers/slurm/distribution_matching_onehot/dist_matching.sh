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


python pretrain.py agent=dist_matching_onehot_augmented env.render_mode=null save_video=false num_train_frames=304000 use_wandb=false agent.T_init_steps=0 configs/env=multiplerooms10_3x3 env.max_steps=${HORIZON} agent.ideal=false agent.data_type=all num_seed_frames=$((3*HORIZON)) agent.update_actor_every_steps=$((3*HORIZON)) agent.epsilon_schedule=${EPS_GREEDY} "agent.sink_schedule='${SINK_SCHEDULE}'" agent.lr_actor=1 agent.pmd_steps=500 agent.window_size=20 agent.data_type=unique agent.unique_window=false agent.n_subsamples=${N_SUBSAMPLES} agent.subsampling_strategy="eder"