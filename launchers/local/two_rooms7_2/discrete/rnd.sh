#!/usr/bin/env bash

CUDA_DEVICE=${1:-0}
python train.py --config-name=train/finetune agent=rnd_discrete env=two_rooms7_2 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd/151047_494131_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=0
python train.py --config-name=train/finetune agent=rnd_discrete env=two_rooms7_2 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd/151047_494131_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=1
python train.py --config-name=train/finetune agent=rnd_discrete env=two_rooms7_2 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd/151047_494131_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=2
python train.py --config-name=train/finetune agent=rnd_discrete env=two_rooms7_2 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd/151047_494131_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=3
python train.py --config-name=train/finetune agent=rnd_discrete env=two_rooms7_2 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd/151047_494131_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=4
python train.py --config-name=train/finetune agent=rnd_discrete env=two_rooms7_2 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd/151047_494131_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=5
python train.py --config-name=train/finetune agent=rnd_discrete env=two_rooms7_2 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/two_rooms/rnd/151047_494131_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=6
