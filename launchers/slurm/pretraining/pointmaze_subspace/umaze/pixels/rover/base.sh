#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
conda activate dist_matching
set -u
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE_ACTOR="${BATCH_SIZE_ACTOR:-80000}"
FEATURE_DIM="${FEATURE_DIM:-1024}"
SUBSAMPLES="${SUBSAMPLES:-10000}"
LAMBDA_REG="${LAMBDA_REG:-1e-3}"
CHOLESKY_TOLERANCE="${CHOLESKY_TOLERANCE:-1e-6}"
SINK_IDX="${SINK_IDX:-0}"
START_X="${START_X:--1.0}"
START_Y="${START_Y:-1.0}"

# Index 0 matches the saved run; index 1 is a lower-sink comparison.
sink_schedules=(
    "linear(0.0,5.0,50000,250000)"
    "linear(0.0,1.0,50000,250000)"
)
SINK_SCHEDULE="${sink_schedules[${SINK_IDX}]}"
DYNAMICS_KERNEL_TYPE="${DYNAMICS_KERNEL_TYPE:-inner_product}"
RUN_LABEL="${DYNAMICS_KERNEL_TYPE}_batch${BATCH_SIZE_ACTOR}_dim${FEATURE_DIM}_nys${SUBSAMPLES}_lambda${LAMBDA_REG}_chol${CHOLESKY_TOLERANCE}_sink${SINK_IDX}_start${START_X}_${START_Y}"

python pretrain_parallel.py \
    --config-name=pretrain_parallel/pointmaze_subspace_pixels \
    env=pointmaze/pointmaze_umaze_full_state_discrete \
    "env.pointmaze.start_position=[${START_X},${START_Y}]" \
    seed="${SEED}" \
    device="${DEVICE}" \
    num_train_frames=250000 \
    eval_every_frames=20000 \
    plot_eval_trajectories=true \
    save_eval_best=true \
    save_snapshot=true \
    'snapshots=[100000,250000]' \
    agent.batch_size_actor="${BATCH_SIZE_ACTOR}" \
    agent.dynamics_kernel_input=pixels \
    agent.dynamics_encoder_mode=trainable \
    agent.dynamics_kernel_type="${DYNAMICS_KERNEL_TYPE}" \
    agent.coverage_kernel_input=pixels \
    agent.coverage_kernel_type=gaussian \
    agent.train_coverage_encoder=false \
    agent.feature_dim="${FEATURE_DIM}" \
    agent.subsamples="${SUBSAMPLES}" \
    agent.pca_truncation=1000 \
    agent.lambda_reg="${LAMBDA_REG}" \
    agent.nystrom_cholesky_tolerance="${CHOLESKY_TOLERANCE}" \
    agent.use_recent_actor_support=true \
    "agent.sink_schedule='${SINK_SCHEDULE}'" \
    snapshot_dir="models/pointmaze_subspace/umaze/pixels/rover/${RUN_LABEL}/seed_${SEED}" \
    use_wandb=true \
    wandb_project=pointmaze_subspace \
    wandb_tag=rover_umaze_pixels \
    wandb_run_name="umaze_pixels_rover_${RUN_LABEL}_seed${SEED}"
