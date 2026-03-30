#!/usr/bin/env bash

CUDA_DEVICE=${1:-0}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=multiplerooms5_4x4_1 num_grad_steps=20000 wandb_tag=random replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/multiplerooms/random seed=0 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=multiplerooms5_4x4_1 num_grad_steps=20000 wandb_tag=random replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/multiplerooms/random seed=1 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=multiplerooms5_4x4_1 num_grad_steps=20000 wandb_tag=random replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/multiplerooms/random seed=2 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=multiplerooms5_4x4_1 num_grad_steps=20000 wandb_tag=random replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/multiplerooms/random seed=3 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=multiplerooms5_4x4_1 num_grad_steps=20000 wandb_tag=random replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/multiplerooms/random seed=4 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=multiplerooms5_4x4_1 num_grad_steps=20000 wandb_tag=random replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/multiplerooms/random seed=5 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=multiplerooms5_4x4_1 num_grad_steps=20000 wandb_tag=random replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/multiplerooms/random seed=6 device=cuda:${CUDA_DEVICE}
