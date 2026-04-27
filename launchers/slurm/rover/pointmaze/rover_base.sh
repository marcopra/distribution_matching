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
    "linear(0.0, 0.001, 200_000)"
    "linear(0.0, 0.01,  200_000)"
    "linear(0.0, 0.1,   200_000)"
    "linear(0.0, 0,     200_000)"
    "linear(0.0, 1.0,   200_000)"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"

    

python pretrain.py \
    --config-name=pretrain/pretrain_pointmaze_umaze_1 \
    agent=rover \
    use_wandb=true \
    agent.batch_size_actor=${BATCH_SIZE_ACTOR} \
    agent.lr_actor=${LR_ACTOR} \
    agent.feature_dim=${FEATURE_DIM}\
    seed=${SEED} \
    "agent.sink_schedule='${SINK_SCHEDULE}'" \
    env=${ENV} \
    agent.update_every_steps=100 \
    agent.pmd_steps=100 \
    num_train_frames=200_000 \


