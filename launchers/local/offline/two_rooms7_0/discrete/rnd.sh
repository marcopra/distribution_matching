#!/usr/bin/env bash

CUDA_DEVICE=${1:-0}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_0 wandb_tag=rnd replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd seed=0 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_0 wandb_tag=rnd replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd seed=1 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_0 wandb_tag=rnd replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd seed=2 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_0 wandb_tag=rnd replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd seed=3 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_0 wandb_tag=rnd replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd seed=4 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_0 wandb_tag=rnd replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd seed=5 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms env=two_rooms7_0 wandb_tag=rnd replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd seed=6 device=cuda:${CUDA_DEVICE}
