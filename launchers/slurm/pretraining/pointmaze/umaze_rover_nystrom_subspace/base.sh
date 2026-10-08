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
    "0.0"
    "linear(0.0,0.8,25000,50000)"
)
SINK_SCHEDULE="${sink_schedules[$((SINK_IDX - 1))]}"

RUN_LABEL="chol${CHOLESKY_TOLERANCE}_bw${KERNEL_BANDWIDTH_MULT}_batch${BATCH_SIZE_ACTOR}_nys${SUBSAMPLE}_pca${PCA_TRUNCATION}_sink${SINK_IDX}"

cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
conda activate dist_matching
export HYDRA_FULL_ERROR=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pretrain_parallel.py \
    --config-name=pretrain_parallel/pretrain_pointmaze_umaze_1_subspace \
    env=pointmaze/pointmaze_umaze_full_state_discrete \
    seed="${SEED}" \
    wandb_project=pointmaze_subspace_hp \
    use_wandb=true \
    num_train_frames=100000 \
    eval_every_frames=100000 \
    agent.lr_actor=100 \
    +coverage_eval_enabled=true \
    +coverage_num_trajectories=50 \
    +coverage_grid_size=90 \
    +coverage_radius=0.08 \
    +coverage_expansion_tolerance=0.25 \
    parallel_attach_eval_env_for_debug=true \
    snapshot_dir="models/pointmaze/umaze/states/rover_subspace_sweep/${RUN_LABEL}/seed_${SEED}" \
    agent.embeddings=true \
    agent.linear_projection=true \
    agent.coverage_embeddings=false \
    agent.coverage_state_filter=2 \
    agent.kernel_type=gaussian \
    agent.kernel_bandwidth=null \
    agent.kernel_bandwidth_mult="${KERNEL_BANDWIDTH_MULT}" \
    agent.coverage_kernel_bandwidth=null \
    agent.coverage_kernel_bandwidth_mult="${KERNEL_BANDWIDTH_MULT}" \
    agent.nystrom_cholesky_tolerance="${CHOLESKY_TOLERANCE}" \
    agent.lambda_reg=1e-3 \
    agent.subsampling_strategy=pivoted_cholesky \
    agent.nystrom_synthetic_subsamples=false \
    agent.nystrom_exact_grid=false \
    agent.debug_fixed_dataset_updates=false \
    agent.subsamples="${SUBSAMPLE}" \
    agent.pca_truncation="${PCA_TRUNCATION}" \
    agent.batch_size_actor="${BATCH_SIZE_ACTOR}" \
    "agent.sink_schedule='${SINK_SCHEDULE}'"
