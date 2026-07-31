#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua-longrun

set -euo pipefail

: "${SEED:?SEED is required}"
: "${KERNEL_BANDWIDTH_MULTIPLIER:?KERNEL_BANDWIDTH_MULTIPLIER is required}"
: "${SUBSAMPLE:?SUBSAMPLE is required}"
: "${BATCH_SIZE_ACTOR:?BATCH_SIZE_ACTOR is required}"
: "${SINK_IDX:?SINK_IDX is required}"

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer: ${SEED}" >&2
    exit 2
fi
if ! [[ "${SUBSAMPLE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SUBSAMPLE must be a positive integer: ${SUBSAMPLE}" >&2
    exit 2
fi
if ! [[ "${BATCH_SIZE_ACTOR}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE_ACTOR must be a positive integer: ${BATCH_SIZE_ACTOR}" >&2
    exit 2
fi
if (( BATCH_SIZE_ACTOR < SUBSAMPLE )); then
    echo "BATCH_SIZE_ACTOR must be >= SUBSAMPLE" >&2
    exit 2
fi
if ! [[ "${SINK_IDX}" =~ ^[0-4]$ ]]; then
    echo "SINK_IDX must be one of 0, 1, 2, 3, 4: ${SINK_IDX}" >&2
    exit 2
fi
if ! awk -v value="${KERNEL_BANDWIDTH_MULTIPLIER}" 'BEGIN { exit !(value + 0 > 0) }'; then
    echo "KERNEL_BANDWIDTH_MULTIPLIER must be positive: ${KERNEL_BANDWIDTH_MULTIPLIER}" >&2
    exit 2
fi

sink_schedules=(
    "linear(0.0, 0.001, 500000)"
    "linear(0.0, 0.01,  500000)"
    "linear(0.0, 0.1,   500000)"
    "linear(0.0, 1.0,   500000)"
    "0.0"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"
RUN_LABEL="bw${KERNEL_BANDWIDTH_MULTIPLIER}_nys${SUBSAMPLE}_batch${BATCH_SIZE_ACTOR}_sink${SINK_IDX}"

cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
conda activate dist_matching
export HYDRA_FULL_ERROR=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pretrain_parallel.py \
    --config-name=pretrain_parallel/pretrain_pointmaze_largedense_nystrom \
    seed="${SEED}" \
    num_train_frames=1_000_000 \
    eval_every_frames=100_000 \
    coverage_eval_enabled=true \
    coverage_num_trajectories=50 \
    coverage_grid_size=90 \
    coverage_radius=0.08 \
    coverage_expansion_tolerance=0.25 \
    plot_eval_trajectories=false \
    use_wandb=true \
    save_snapshot=true \
    "snapshots=[100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]" \
    snapshot_dir="models/pointmaze/largedense/states/rover_nystrom_sweep/${RUN_LABEL}/seed_${SEED}" \
    wandb_project=pretrain_pointmaze_parallel \
    wandb_tag="rover_nystrom_largedense_online_${RUN_LABEL}" \
    wandb_run_name="largedense_${RUN_LABEL}_seed${SEED}" \
    agent.embeddings=false \
    agent.debug_fixed_dataset_updates=false \
    agent.nystrom_synthetic_subsamples=false \
    agent.nystrom_exact_grid=false \
    agent.use_encoded_fifo=true \
    agent.subsampling_strategy=random \
    agent.subsamples="${SUBSAMPLE}" \
    agent.batch_size_actor="${BATCH_SIZE_ACTOR}" \
    agent.kernel_bandwidth=null \
    agent.kernel_bandwidth_mult="${KERNEL_BANDWIDTH_MULTIPLIER}" \
    "agent.sink_schedule='${SINK_SCHEDULE}'"
