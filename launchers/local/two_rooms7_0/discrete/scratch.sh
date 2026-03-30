#!/usr/bin/env bash

CUDA_DEVICE=${1:-0}
python train.py --config-name=train/finetune env=two_rooms7_0 device=cuda:${CUDA_DEVICE} wandb_tag=from_scratch num_train_frames=50_000 seed=0
python train.py --config-name=train/finetune env=two_rooms7_0 device=cuda:${CUDA_DEVICE} wandb_tag=from_scratch num_train_frames=50_000 seed=1
python train.py --config-name=train/finetune env=two_rooms7_0 device=cuda:${CUDA_DEVICE} wandb_tag=from_scratch num_train_frames=50_000 seed=2
python train.py --config-name=train/finetune env=two_rooms7_0 device=cuda:${CUDA_DEVICE} wandb_tag=from_scratch num_train_frames=50_000 seed=3
python train.py --config-name=train/finetune env=two_rooms7_0 device=cuda:${CUDA_DEVICE} wandb_tag=from_scratch num_train_frames=50_000 seed=4
python train.py --config-name=train/finetune env=two_rooms7_0 device=cuda:${CUDA_DEVICE} wandb_tag=from_scratch num_train_frames=50_000 seed=5
python train.py --config-name=train/finetune env=two_rooms7_0 device=cuda:${CUDA_DEVICE} wandb_tag=from_scratch num_train_frames=50_000 seed=6
