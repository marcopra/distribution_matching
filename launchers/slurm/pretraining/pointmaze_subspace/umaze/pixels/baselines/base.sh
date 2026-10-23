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
AGENT="${AGENT:-rnd_discrete}"
START_X="${START_X:--1.0}"
START_Y="${START_Y:-1.0}"
START_LABEL="start_${START_X}_${START_Y}"

python pretrain.py \
    --config-name=pretrain/pointmaze_subspace_baselines \
    env=pointmaze/pointmaze_umaze_full_state_discrete \
    "env.pointmaze.start_position=[${START_X},${START_Y}]" \
    env.continuing_task=false \
    obs_type=pixels \
    frame_stack=4 \
    agent="${AGENT}" \
    seed="${SEED}" \
    device="${DEVICE}" \
    num_train_frames=1000000 \
    eval_every_frames=20000 \
    plot_eval_trajectories=true \
    save_eval_best=true \
    save_snapshot=false \
    'snapshots=[250000,500000,750000,1000000]' \
    snapshot_dir="models/pointmaze_subspace/umaze/pixels/${AGENT}/${START_LABEL}/seed_${SEED}" \
    use_wandb=true \
    wandb_project=pointmaze_subspace \
    wandb_tag="${AGENT}_umaze_pixels_${START_LABEL}" \
    wandb_run_name="umaze_pixels_${AGENT}_${START_LABEL}_seed${SEED}"
