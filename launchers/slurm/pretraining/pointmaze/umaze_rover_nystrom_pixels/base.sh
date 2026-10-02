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
    "linear(0.0,1.0,25000,50000)"
)
SINK_SCHEDULE="${sink_schedules[$((SINK_IDX - 1))]}"

KERNEL_BANDWIDTH_MULT="${KERNEL_BANDWIDTH_MULT:-1.0}"
PRETRAINED_ENCODER_PATH="${PRETRAINED_ENCODER_PATH:-/home/mprattico-iit.local/distribution_matching/tmp_models/pretrained/encoder.pt}"

case "${ENCODER_MODE}" in
    scratch)
        P_PATH=none
        FREEZE_ENCODER=false
        ;;
    pretrained_finetune)
        P_PATH="${PRETRAINED_ENCODER_PATH}"
        FREEZE_ENCODER=false
        ;;
    pretrained_frozen)
        P_PATH="${PRETRAINED_ENCODER_PATH}"
        FREEZE_ENCODER=true
        ;;
    *)
        echo "Unknown ENCODER_MODE: ${ENCODER_MODE}" >&2
        exit 2
        ;;
esac

case "${KERNEL_TYPE}" in
    inner_product)
        WHITEN_REPRESENTATIONS=false
        NYSTROM_CHOLESKY_TOLERANCE="${CHOLESKY_TOLERANCE}"
        ;;
    gaussian)
        WHITEN_REPRESENTATIONS=true
        NYSTROM_CHOLESKY_TOLERANCE=0
        ;;
    *)
        echo "Unknown KERNEL_TYPE: ${KERNEL_TYPE}" >&2
        exit 2
        ;;
esac

RUN_LABEL="${KERNEL_TYPE}_${ENCODER_MODE}_feat${FEATURE_DIM}_${FEATURE_MODE}_nys${SUBSAMPLE}_pca${PCA_TRUNCATION}_batch${BATCH_SIZE_ACTOR}_sink${SINK_IDX}_lambda${LAMBDA_REG}_bw${KERNEL_BANDWIDTH_MULT}_chol${NYSTROM_CHOLESKY_TOLERANCE}"

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
    agent.lr_actor=10000 \
    agent.kernel_type="${KERNEL_TYPE}" \
    agent.kernel_bandwidth_mult="${KERNEL_BANDWIDTH_MULT}" \
    num_train_frames=100000 \
    agent.update_actor_every_steps=2000 \
    eval_every_frames=25_000 \
    +coverage_eval_enabled=true \
    +coverage_num_trajectories=50 \
    +coverage_grid_size=90 \
    +coverage_radius=0.08 \
    +coverage_expansion_tolerance=0.25 \
    +plot_eval_trajectories=false \
    save_eval_best=true \
    save_snapshot=false \
    use_wandb=true \
    wandb_project=pointmaze_hp \
    agent.feature_dim="${FEATURE_DIM}" \
    agent.mode="${FEATURE_MODE}" \
    agent.whiten_representations="${WHITEN_REPRESENTATIONS}" \
    agent.freeze_encoder="${FREEZE_ENCODER}" \
    p_path="${P_PATH}" \
    agent.nystrom_cholesky_tolerance="${NYSTROM_CHOLESKY_TOLERANCE}" \
    agent.lambda_reg="${LAMBDA_REG}" \
    agent.subsampling_strategy=pivoted_cholesky \
    agent.debug_fixed_dataset_updates=false \
    agent.nystrom_synthetic_subsamples=false \
    agent.nystrom_exact_grid=false \
    agent.subsamples="${SUBSAMPLE}" \
    agent.pca_truncation="${PCA_TRUNCATION}" \
    agent.batch_size_actor="${BATCH_SIZE_ACTOR}" \
    "agent.sink_schedule='${SINK_SCHEDULE}'" \
    agent.linear_projection=true \
    snapshot_dir="models/pixels/gym/dist_matching_umaze_sweep/${RUN_LABEL}/seed_${SEED}" \
    wandb_tag="rover_nystrom_umaze_pixels_${RUN_LABEL}" \
    wandb_run_name="umaze_pixels_${RUN_LABEL}_seed${SEED}"
