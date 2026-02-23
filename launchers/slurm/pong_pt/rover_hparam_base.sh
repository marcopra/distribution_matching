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

# Decode sink_schedule from index to avoid quoting issues in sbatch --export
sink_schedules=(
    "linear(0.0, 0.0001, 1_000_000)"
    "linear(0.0, 1, 1_000_000)"
    "linear(0.0, 0.1, 100_000)"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"

python pretrain.py \
    use_wandb=true \
    wandb_project="rover_pong" \
    wandb_tag="rover_hparam" \
    agent.lr_actor=${LR_ACTOR} \
    agent.pmd_steps=500 \
    eval_every_frames=100_000 \
    num_train_frames=1_100_000 \
    agent.update_every_steps=5 \
    agent.update_actor_every_steps=10000 \
    env=pong \
    device=cuda \
    seed=${SEED} \
    save_video=false \
    env.render_mode=rgb_array \
    agent.batch_size_actor=${BATCH_SIZE_ACTOR} \
    agent.curl=false \
    agent.feature_dim=${FEATURE_DIM} \
    "agent.sink_schedule='${SINK_SCHEDULE}'"
