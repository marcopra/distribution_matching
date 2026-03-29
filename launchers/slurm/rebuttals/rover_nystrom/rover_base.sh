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
# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> linear(0.0, 0.0, 50000)
#   1 -> linear(0.0, 0.005,  50000)
#   2 -> linear(0.0, 1,      50000)
#   3 -> linear(0.1, 0.1,    50_000)


sink_schedules=(
    "linear(0.0, 0.0, 50000)"
    "linear(0.0, 0.005,  50000)"
    "linear(0.0, 1,      50000)"
    "linear(0.1, 0.1,    50_000)"
)

SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"


python pretrain.py \
    --config-name=pretrain/pretrain_rover_multiplerooms \
    agent=rover_nystrom \
    use_wandb=true \
    wandb_project="rover_nystrom" \
    agent.lr_actor=${LR_ACTOR} \
    agent.pmd_steps=50 \
    num_train_frames=50000 \
    num_seed_frames=1000 \
    agent.update_actor_every_steps=900 \
    agent.subsamples="${SUBSAMPLES}" \
    device=cuda \
    seed=${SEED} \
    save_video=false \
    agent.batch_size_actor=${BATCH_SIZE_ACTOR} \
    "agent.sink_schedule='${SINK_SCHEDULE}'" \
    replay_buffer_size=1_000_000 \
    agent.pmd_eta_mode=backtracking \
    env=${ENV} \

