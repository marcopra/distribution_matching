#!/usr/bin/env bash

CUDA_DEVICE=${1:-0}
python train.py --config-name=train/finetune agent=ddpg_discrete env=multiplerooms5_4x4_1 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/multiplerooms/rnd/151038_355222_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=0
python train.py --config-name=train/finetune agent=ddpg_discrete env=multiplerooms5_4x4_1 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/multiplerooms/rnd/151038_355222_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=1
python train.py --config-name=train/finetune agent=ddpg_discrete env=multiplerooms5_4x4_1 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/multiplerooms/rnd/151038_355222_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=2
python train.py --config-name=train/finetune agent=ddpg_discrete env=multiplerooms5_4x4_1 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/multiplerooms/rnd/151038_355222_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=3
python train.py --config-name=train/finetune agent=ddpg_discrete env=multiplerooms5_4x4_1 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/multiplerooms/rnd/151038_355222_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=4
python train.py --config-name=train/finetune agent=ddpg_discrete env=multiplerooms5_4x4_1 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/multiplerooms/rnd/151038_355222_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=5
python train.py --config-name=train/finetune agent=ddpg_discrete env=multiplerooms5_4x4_1 device=cuda:${CUDA_DEVICE} wandb_tag=rnd p_path=/home/mprattico/distribution_matching/models/discrete/multiplerooms/rnd/151038_355222_rnd/models/discrete/gym/rnd/1/snapshot.pt seed=6
