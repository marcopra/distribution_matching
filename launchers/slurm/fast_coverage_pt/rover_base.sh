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

python pretrain.py --config-name=pretrain/pretrain_rover_multiplerooms \
    env=${ENV} \
    obs_type=${OBS_TYPE} \
    agent.feature_dim=200 \
    "agent.sink_schedule='linear(0.0, 0.5, 100_000)'" \
    discount=0.9  \
    agent.update_every_steps=50 \
    agent.batch_size_actor=8000 \
    agent.lr_actor=200 \
    agent.pmd_steps=50 \
    num_seed_frames=1000 \
    agent.update_actor_every_steps=1500 \
    num_train_frames=110000 \
    seed=${SEED}
