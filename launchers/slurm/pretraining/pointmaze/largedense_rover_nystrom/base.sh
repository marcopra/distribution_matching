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



sink_schedules=(
    "linear(0.0, 0.001, 500000)"
    "linear(0.0, 0.01,  500000)"
    "linear(0.0, 0.1,   500000)"
    "linear(0.0, 0.8,   500000)"
    "0.8"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"

kernel_bandwidth_schedules=(
    "linear(0.05, 0.3,  500000)"
    "linear(0.1,  0.25, 500000)"
    "linear(0.05,  0.2, 500000)"
    "linear(0.1,  0.2, 500000)"
    "linear(0.05,  0.15, 500000)"
    "linear(0.01,  0.18, 500000)"
)
KERNEL_BANDWIDTH_SCHEDULE="${kernel_bandwidth_schedules[$KERNEL_BANDWIDTH_IDX]}"
RUN_LABEL="bw${KERNEL_BANDWIDTH_IDX}_nys${SUBSAMPLE}_batch${BATCH_SIZE_ACTOR}_sink${SINK_IDX}"

cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
conda activate dist_matching
export HYDRA_FULL_ERROR=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pretrain_parallel.py \
    --config-name=pretrain_parallel/pretrain_pointmaze_largedense_nystrom \
    seed="${SEED}" \
    num_train_frames=5000000 \
    eval_every_frames=50_000 \
    coverage_eval_enabled=true \
    coverage_num_trajectories=50 \
    coverage_grid_size=90 \
    coverage_radius=0.08 \
    coverage_expansion_tolerance=0.25 \
    plot_eval_trajectories=false \
    use_wandb=true \
    snapshot_dir="models/pointmaze/largedense/states/rover_nystrom_sweep/${RUN_LABEL}/seed_${SEED}" \
    wandb_project=pointmaze_hp \
    wandb_tag="rover_nystrom_largedense_online_${RUN_LABEL}" \
    wandb_run_name="largedense_${RUN_LABEL}_seed${SEED}" \
    agent.embeddings=false \
    agent.lambda_reg=1e-6 \
    agent.subsampling_strategy=pivoted_cholesky \
    agent.debug_fixed_dataset_updates=false \
    agent.nystrom_synthetic_subsamples=false \
    agent.nystrom_exact_grid=false \
    agent.subsamples="${SUBSAMPLE}" \
    agent.batch_size_actor="${BATCH_SIZE_ACTOR}" \
    "agent.kernel_bandwidth='${KERNEL_BANDWIDTH_SCHEDULE}'" \
    agent.kernel_bandwidth_mult=null \
    "agent.sink_schedule='${SINK_SCHEDULE}'"
