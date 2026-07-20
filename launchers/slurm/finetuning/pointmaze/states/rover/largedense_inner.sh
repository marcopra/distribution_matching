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

python train.py --config-name=train/finetune_maze \
    agent=ddpg_discrete_with_kernel_actor \
    env="${ENV}" \
    p_path="${MODEL_PATH}" \
    agent.feature_dim=256 \
    agent.dataset_dim=16000 \
    agent.actor_lr="${ACTOR_LR}" \
    seed="${SEED}" \
    obs_type=states \
    num_train_frames=500000 \
    num_seed_frames=20000 \
    update_actor_after_critic_steps=10000 \
    use_wandb=true \
    wandb_project=pointmaze_ft \
    wandb_tag=rover_largedense \
    hydra.run.dir="exp_local/pointmaze_ft/${SLURM_JOB_ID}" \
    device=cuda
