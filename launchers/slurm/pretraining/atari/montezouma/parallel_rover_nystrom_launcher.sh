#!/bin/bash

seeds="1 2 3"

batch_sizes_actor=(
    # 50000
    100000
    # 200000
)

subsamples=(
    # 5000
    10000
    # 20000
)

for seed in $seeds; do
    for batch_size_actor in "${batch_sizes_actor[@]}"; do
        for subsample in "${subsamples[@]}"; do
            sbatch --export=SEED=$seed,BATCH_SIZE_ACTOR=$batch_size_actor,SUBSAMPLE=$subsample launchers/slurm/pretraining/atari/montezouma/parallel_rover_nystrom_base.sh
        done
    done
done
