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



python pretrain.py \
    --config-name=pretrain/pretrain_montezouma \
    agent=rover_nystrom \
    use_wandb=true \
    obs_type="pixels" \
    agent.lr_actor=${LR_ACTOR} \
    agent.subsamples="${SUBSAMPLES}" \
    device=cuda \
    seed=${SEED} \
    save_video=false \
    agent.batch_size_actor=${BATCH_SIZE_ACTOR} \
    "agent.sink_schedule='${SINK_SCHEDULE}'" \
    replay_buffer_size=1_000_000 \
    agent.pmd_eta_mode=backtracking \
    env=${ENV} \

