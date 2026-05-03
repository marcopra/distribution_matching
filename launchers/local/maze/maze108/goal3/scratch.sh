CUDA_DEVICE=${1:-0}

python train.py --config-name=train/finetune_maze agent=ddpg_discrete seed=0 env=gridworld/maze_108_seed7_env_3 agent.actor_lr=1e-4 obs_type=pixels device=cuda:${CUDA_DEVICE} agent.actor_lr=1e-4; 
python train.py --config-name=train/finetune_maze agent=ddpg_discrete seed=1 env=gridworld/maze_108_seed7_env_3 agent.actor_lr=1e-4 obs_type=pixels device=cuda:${CUDA_DEVICE} agent.actor_lr=1e-4; 
python train.py --config-name=train/finetune_maze agent=ddpg_discrete seed=2 env=gridworld/maze_108_seed7_env_3 agent.actor_lr=1e-4 obs_type=pixels device=cuda:${CUDA_DEVICE} agent.actor_lr=1e-4; 
python train.py --config-name=train/finetune_maze agent=ddpg_discrete seed=3 env=gridworld/maze_108_seed7_env_3 agent.actor_lr=1e-4 obs_type=pixels device=cuda:${CUDA_DEVICE} agent.actor_lr=1e-4; 
python train.py --config-name=train/finetune_maze agent=ddpg_discrete seed=4 env=gridworld/maze_108_seed7_env_3 agent.actor_lr=1e-4 obs_type=pixels device=cuda:${CUDA_DEVICE} agent.actor_lr=1e-4; 
python train.py --config-name=train/finetune_maze agent=ddpg_discrete seed=5 env=gridworld/maze_108_seed7_env_3 agent.actor_lr=1e-4 obs_type=pixels device=cuda:${CUDA_DEVICE} agent.actor_lr=1e-4; 
python train.py --config-name=train/finetune_maze agent=ddpg_discrete seed=6 env=gridworld/maze_108_seed7_env_3 agent.actor_lr=1e-4 obs_type=pixels device=cuda:${CUDA_DEVICE} agent.actor_lr=1e-4; 

