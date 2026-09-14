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
    "linear(0.0, 1,   500000)"
    "linear(0.0, 0.8,   500000)"
    "0.8"
    "linear(0.0, 0.1,   500000)"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"

kernel_bandwidth_schedules=(
    "0.35"
    "0.3"
    "0.25"
    "0.2"
    "0.1"
    "0.15"
)
KERNEL_BANDWIDTH_SCHEDULE="${kernel_bandwidth_schedules[$KERNEL_BANDWIDTH_IDX]}"
RUN_LABEL="bw${KERNEL_BANDWIDTH_IDX}_feat${FEATURE_DIM}_${FEATURE_MODE}_nys${SUBSAMPLE}_batch${BATCH_SIZE_ACTOR}_sink${SINK_IDX}_lambda${LAMBDA_REG}"

cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
conda activate dist_matching
export HYDRA_FULL_ERROR=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pretrain_parallel.py \
    --config-name=pretrain_parallel/pretrain_pointmaze_umaze_1_pixels \
    env=pointmaze/pointmaze_umaze_goal_1 \
    obs_type=pixels \
    agent.embeddings=true \
    seed="${SEED}" \
    agent.lr_actor=1000 \
    num_train_frames=1000000 \
    eval_every_frames=10_000 \
    +coverage_eval_enabled=true \
    +coverage_num_trajectories=50 \
    +coverage_grid_size=90 \
    +coverage_radius=0.08 \
    +coverage_expansion_tolerance=0.25 \
    +plot_eval_trajectories=false \
    save_eval_best=true \
    save_snapshot=false \
    snapshots=[] \
    use_wandb=true \
    snapshot_dir="models/pointmaze/umaze_goal_1/pixels/rover_nystrom_sweep/${RUN_LABEL}/seed_${SEED}" \
    wandb_project=pointmaze_hp \
    wandb_tag="rover_nystrom_umaze_goal_1_pixels_online_${RUN_LABEL}" \
    wandb_run_name="umaze_goal_1_pixels_${RUN_LABEL}_seed${SEED}" \
    agent.feature_dim="${FEATURE_DIM}" \
    agent.mode="${FEATURE_MODE}" \
    agent.whiten_representations=true \
    agent.whitening_variance=0.99 \
    agent.whitening_components="${FEATURE_DIM}" \
    agent.whitening_epsilon=1e-5 \
    agent.whitening_unit_trace=true \
    agent.nystrom_cholesky_tolerance=0 \
    agent.lambda_reg="${LAMBDA_REG}" \
    agent.subsampling_strategy=pivoted_cholesky \
    agent.debug_fixed_dataset_updates=false \
    agent.nystrom_synthetic_subsamples=false \
    agent.nystrom_exact_grid=false \
    agent.subsamples="${SUBSAMPLE}" \
    agent.pca_truncation="${SUBSAMPLE}" \
    agent.batch_size_actor="${BATCH_SIZE_ACTOR}" \
    "agent.kernel_bandwidth='${KERNEL_BANDWIDTH_SCHEDULE}'" \
    agent.kernel_bandwidth_mult=null \
    "agent.sink_schedule='${SINK_SCHEDULE}'"
