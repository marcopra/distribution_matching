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
    "linear(0.0, 0.0001, 1_000_000)"
    "linear(0.0, 0.001,  2_000_000)"
    "linear(0.0, 1,      1_000_000)"
    "linear(0.0, 0.0001, 500_000)"
    "linear(1.0, 1.0,    100_000)"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pretrain.py \
    --config-name=pretrain/pretrain_montezouma \
    use_wandb=true \
    seed=${SEED} \
    agent.batch_size_actor=${BATCH_SIZE_ACTOR} \
    agent.subsamples=${SUBSAMPLE} \
    agent.sink_schedule=0.0 \
    num_train_frames=1_500_000

