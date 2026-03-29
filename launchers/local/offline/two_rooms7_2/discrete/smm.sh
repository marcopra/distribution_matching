#!/usr/bin/env bash

CUDA_DEVICE=${1:-0}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_2 wandb_tag=smm replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/two_rooms/smm seed=0 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_2 wandb_tag=smm replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/two_rooms/smm seed=1 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_2 wandb_tag=smm replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/two_rooms/smm seed=2 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_2 wandb_tag=smm replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/two_rooms/smm seed=3 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_2 wandb_tag=smm replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/two_rooms/smm seed=4 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_2 wandb_tag=smm replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/two_rooms/smm seed=5 # device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_2 wandb_tag=smm replay_buffer_dir=/home/mprattico-iit.local/distribution_matching/models/discrete/two_rooms/smm seed=6 # device=cuda:${CUDA_DEVICE}


