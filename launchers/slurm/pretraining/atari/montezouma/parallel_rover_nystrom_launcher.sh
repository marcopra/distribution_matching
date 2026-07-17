#!/bin/bash

seeds="1"

batch_sizes_actor=(
    50000
    100000
    # 200000
)

subsamples=(
    5000
    10000
    # 20000
)

feature_dims=(
    64
    256
    # 512
    # 1024
)
for seed in $seeds; do
    for batch_size_actor in "${batch_sizes_actor[@]}"; do
        for subsample in "${subsamples[@]}"; do
            for feature_dim in "${feature_dims[@]}"; do
                sbatch --export=SEED=$seed,BATCH_SIZE_ACTOR=$batch_size_actor,SUBSAMPLE=$subsample,FEATURE_DIM=$feature_dim launchers/slurm/pretraining/atari/montezouma/parallel_rover_nystrom_base.sh
            done
        done
    done
done
