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


# Decode sink_schedule from index to avoid quoting issues in sbatch --export
sink_schedules=(
    "linear(0.0,0.5,100_000)"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"


python pretrain.py --config-name=pretrain/pretrain_rover_multiplerooms \
    agent=rover \
    obs_type=pixels \
    env=${ENV} \
    agent.embeddings=true \
    discount=${DISCOUNT} \
    eval_every_frames=50000 \
    num_eval_episodes=20 \
    num_train_frames=160_000 \
    agent.pmd_steps=100 \
    agent.feature_dim=${FEATURE_DIM} \
    agent.lr_actor=100 \
    agent.lr_encoder=1e-3 \
    agent.lr_T=1e-3 \
    "agent.sink_schedule='${SINK_SCHEDULE}'" \
    seed=${SEED}
