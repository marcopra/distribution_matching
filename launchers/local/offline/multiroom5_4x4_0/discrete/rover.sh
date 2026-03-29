#!/usr/bin/env bash

CUDA_DEVICE=${1:-0}

python train_offline.py --config-name=train/train_offline_rooms env=multiplerooms5_4x4_0 num_grad_steps=50000 wandb_tag=rover replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/multiplerooms/rover seed=0 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=multiplerooms5_4x4_0 num_grad_steps=50000 wandb_tag=rover replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/multiplerooms/rover seed=1 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=multiplerooms5_4x4_0 num_grad_steps=50000 wandb_tag=rover replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/multiplerooms/rover seed=2 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=multiplerooms5_4x4_0 num_grad_steps=50000 wandb_tag=rover replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/multiplerooms/rover seed=3 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=multiplerooms5_4x4_0 num_grad_steps=50000 wandb_tag=rover replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/multiplerooms/rover seed=4 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=multiplerooms5_4x4_0 num_grad_steps=50000 wandb_tag=rover replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/multiplerooms/rover seed=5 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=multiplerooms5_4x4_0 num_grad_steps=50000 wandb_tag=rover replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/multiplerooms/rover seed=6 # device=cuda:${CUDA_DEVICE}
