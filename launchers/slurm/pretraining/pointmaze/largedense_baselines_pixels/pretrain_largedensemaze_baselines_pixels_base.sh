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

cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
conda activate dist_matching
export HYDRA_FULL_ERROR=1

DEVICE="${DEVICE:-cuda}"

python pretrain.py \
    --config-name=pretrain/pretrain_umaze_baselines \
    env=pointmaze/pointmaze_largedense_goal_1 \
    obs_type=pixels \
    grayscale=true \
    agent="${AGENT}" \
    seed="${SEED}" \
    device="${DEVICE}" \
    num_train_frames=1000000 \
    eval_every_frames=50000 \
    coverage_eval_enabled=true \
    coverage_num_trajectories=50 \
    coverage_grid_size=90 \
    coverage_radius=0.08 \
    coverage_expansion_tolerance=0.25 \
    plot_eval_trajectories=false \
    save_eval_best=true \
    save_snapshot=false \
    snapshots=[] \
    snapshot_dir="models/pointmaze/largedense/pixels/baselines/${AGENT}/seed_${SEED}" \
    wandb_project=pointmaze_baselines \
    wandb_tag="${AGENT}_largedense_pixels" \
    wandb_run_name="largedense_pixels_${AGENT}_seed${SEED}"
