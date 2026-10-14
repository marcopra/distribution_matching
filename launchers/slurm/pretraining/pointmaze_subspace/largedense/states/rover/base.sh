#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
conda activate dist_matching
set -u
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE_ACTOR="${BATCH_SIZE_ACTOR:-16000}"
FEATURE_DIM="${FEATURE_DIM:-256}"
SUBSAMPLES="${SUBSAMPLES:-4000}"
SINK_IDX="${SINK_IDX:-0}"

# Index 0 = reference schedule; index 1 = alternative schedule.
sink_schedules=(
    "linear(0.0,0.8,250000)"
    "linear(0.0,1.0,300000,750000)"
)
SINK_SCHEDULE="${sink_schedules[${SINK_IDX}]}"
RUN_LABEL="batch${BATCH_SIZE_ACTOR}_dim${FEATURE_DIM}_nys${SUBSAMPLES}_sink${SINK_IDX}"

python pretrain_parallel.py \
    --config-name=pretrain_parallel/pointmaze_subspace_states \
    env=pointmaze/pointmaze_largedense_full_state_discrete \
    'env.pointmaze.start_position=[-4.0,3.0]' \
    seed="${SEED}" \
    device="${DEVICE}" \
    num_train_frames=1000000 \
    eval_every_frames=20000 \
    plot_eval_trajectories=true \
    save_eval_best=true \
    save_snapshot=false \
    'snapshots=[250000,500000,750000,1000000]' \
    agent.batch_size_actor="${BATCH_SIZE_ACTOR}" \
    agent.dynamics_kernel_input=states \
    agent.dynamics_encoder_mode=trainable \
    agent.dynamics_kernel_type=inner_product \
    agent.coverage_kernel_input=states \
    agent.coverage_kernel_type=gaussian \
    agent.train_coverage_encoder=false \
    agent.feature_dim="${FEATURE_DIM}" \
    agent.subsamples="${SUBSAMPLES}" \
    agent.pca_truncation="${SUBSAMPLES}" \
    "agent.sink_schedule='${SINK_SCHEDULE}'" \
    snapshot_dir="models/pointmaze_subspace/largedense/states/rover/${RUN_LABEL}/seed_${SEED}" \
    use_wandb=true \
    wandb_project=pointmaze_subspace \
    wandb_tag=rover_largedense_states \
    wandb_run_name="largedense_states_rover_${RUN_LABEL}_seed${SEED}"
