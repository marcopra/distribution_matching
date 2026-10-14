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
AGENT="${AGENT:-rnd_discrete}"

python pretrain.py \
    --config-name=pretrain/pointmaze_subspace_baselines \
    env=pointmaze/pointmaze_umaze_full_state_discrete \
    'env.pointmaze.start_position=[-1.0,1.0]' \
    env.continuing_task=true \
    obs_type=states \
    frame_stack=1 \
    agent="${AGENT}" \
    seed="${SEED}" \
    device="${DEVICE}" \
    num_train_frames=1000000 \
    eval_every_frames=20000 \
    plot_eval_trajectories=true \
    save_eval_best=true \
    save_snapshot=false \
    'snapshots=[250000,500000,750000,1000000]' \
    snapshot_dir="models/pointmaze_subspace/umaze/states/${AGENT}/seed_${SEED}" \
    use_wandb=true \
    wandb_project=pointmaze_subspace \
    wandb_tag="${AGENT}_umaze_states" \
    wandb_run_name="umaze_states_${AGENT}_seed${SEED}"
