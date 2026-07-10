CUDA_DEVICE=${1:-0}

python train.py --config-name=train/finetune_maze agent=cic_discrete p_path="/home/mprattico/distribution_matching/models/maze/108/pixels/cic/snapshot.pt" seed=0 env=gridworld/maze_108_seed7_env agent.actor_lr=1e-4obs_type=pixels device=cuda:${CUDA_DEVICE}; 
python train.py --config-name=train/finetune_maze agent=cic_discrete p_path="/home/mprattico/distribution_matching/models/maze/108/pixels/cic/snapshot.pt" seed=1 env=gridworld/maze_108_seed7_env agent.actor_lr=1e-4obs_type=pixels device=cuda:${CUDA_DEVICE}; 
python train.py --config-name=train/finetune_maze agent=cic_discrete p_path="/home/mprattico/distribution_matching/models/maze/108/pixels/cic/snapshot.pt" seed=2 env=gridworld/maze_108_seed7_env agent.actor_lr=1e-4obs_type=pixels device=cuda:${CUDA_DEVICE}; 
python train.py --config-name=train/finetune_maze agent=cic_discrete p_path="/home/mprattico/distribution_matching/models/maze/108/pixels/cic/snapshot.pt" seed=3 env=gridworld/maze_108_seed7_env agent.actor_lr=1e-4obs_type=pixels device=cuda:${CUDA_DEVICE}; 
python train.py --config-name=train/finetune_maze agent=cic_discrete p_path="/home/mprattico/distribution_matching/models/maze/108/pixels/cic/snapshot.pt" seed=4 env=gridworld/maze_108_seed7_env agent.actor_lr=1e-4obs_type=pixels device=cuda:${CUDA_DEVICE}; 
python train.py --config-name=train/finetune_maze agent=cic_discrete p_path="/home/mprattico/distribution_matching/models/maze/108/pixels/cic/snapshot.pt" seed=5 env=gridworld/maze_108_seed7_env agent.actor_lr=1e-4obs_type=pixels device=cuda:${CUDA_DEVICE}; 
python train.py --config-name=train/finetune_maze agent=cic_discrete p_path="/home/mprattico/distribution_matching/models/maze/108/pixels/cic/snapshot.pt" seed=6 env=gridworld/maze_108_seed7_env agent.actor_lr=1e-4obs_type=pixels device=cuda:${CUDA_DEVICE}; 

