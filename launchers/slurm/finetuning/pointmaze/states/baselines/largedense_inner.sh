#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate dist_matching
export HYDRA_FULL_ERROR=1

python train_parallel.py --config-name=train_parallel/finetune_maze \
    agent="${AGENT}" \
    env="${ENV}" \
    +env.continuing_task=false \
    env.pointmaze.reward_type=sparse \
    p_path="${MODEL_PATH}" \
    agent.actor_lr=1e-4 \
    seed="${SEED}" \
    obs_type=states \
    num_train_frames=1000000 \
    num_seed_frames=90000 \
    num_envs=10 \
    update_actor_after_critic_steps="${ACTOR_UPDATE_THRESHOLD}" \
    use_wandb=true \
    wandb_project=pointmaze_ft \
    wandb_tag="baseline_largedense_${AGENT}" \
    hydra.run.dir="exp_local/pointmaze_ft/${SLURM_JOB_ID}" \
    device=cuda
