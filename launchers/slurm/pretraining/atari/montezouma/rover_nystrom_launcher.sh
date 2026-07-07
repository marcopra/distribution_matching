#!/bin/bash

seeds="1"

batch_sizes_actor=(
    50000
    100000
)

subsamples=(
    5000
    10000
)

for seed in $seeds; do
    for batch_size_actor in "${batch_sizes_actor[@]}"; do
        for subsample in "${subsamples[@]}"; do
                      sbatch --export=SEED=$seed,BATCH_SIZE_ACTOR=$batch_size_actor,SUBSAMPLE=$subsample launchers/slurm/pretraining/atari/montezouma/rover_nystrom_base.sh
        done
    done
done
