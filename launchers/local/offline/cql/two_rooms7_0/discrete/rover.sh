#!/usr/bin/env bash

CUDA_DEVICE=${1:-0}

python train_offline.py --config-name=train/train_offline_rooms agent=cql env=two_rooms7_0 wandb_tag=rover replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rover seed=0 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=two_rooms7_0 wandb_tag=rover replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rover seed=1 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=two_rooms7_0 wandb_tag=rover replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rover seed=2 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=two_rooms7_0 wandb_tag=rover replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rover seed=3 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=two_rooms7_0 wandb_tag=rover replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rover seed=4 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=two_rooms7_0 wandb_tag=rover replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rover seed=5 device=cuda:${CUDA_DEVICE}
python train_offline.py --config-name=train/train_offline_rooms agent=cql env=two_rooms7_0 wandb_tag=rover replay_buffer_dir=/home/mprattico/distribution_matching/models/discrete/two_rooms/rover seed=6 device=cuda:${CUDA_DEVICE}
