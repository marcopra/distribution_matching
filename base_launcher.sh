#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua-longrun

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate dist_matching


export HYDRA_FULL_ERROR=1

# python pretrain.py use_wandb=true wandb_project=rover_pong wandb_tag=rover_hparam2 agent.lr_actor=100 agent.pmd_steps=250 eval_every_frames=10_000 num_train_frames=5_000_000 agent.T_init_steps=10000 agent.update_every_steps=5 agent.update_actor_every_steps=5000 env=pong device=cuda seed=1 save_video=false env.render_mode=rgb_array agent.batch_size_actor=5000 agent.curl=false agent.feature_dim=512 agent.hidden_dim=2048 "agent.sink_schedule='linear(0.0, 0.0, 100_000)'" replay_buffer_size=1_000_000 agent.pmd_eta_mode=backtracking env=pong_score_masked agent.mode=l1 agent.lr_encoder=1e-4
python pretrain.py use_wandb=true wandb_project=rover_pong wandb_tag=rover_hparam2 agent.lr_actor=100 agent.pmd_steps=250 eval_every_frames=10_000 num_train_frames=5_000_000 agent.T_init_steps=10000 agent.update_every_steps=5 agent.update_actor_every_steps=5000 env=pong device=cuda seed=0 save_video=false env.render_mode=rgb_array agent.batch_size_actor=5000 agent.curl=true agent.feature_dim=512 agent.hidden_dim=2048 "agent.sink_schedule='linear(0.0, 0.0, 100_000)'" replay_buffer_size=1_000_000 agent.pmd_eta_mode=backtracking env=pong_score_masked agent.mode=l1 agent.lr_encoder=1e-4 seed=11