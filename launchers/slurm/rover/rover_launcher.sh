#!/bin/bash

seeds="1"
envs=(
"pong"
)
batch_sizes_actor=(
5000
8100
15000
20000
)


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for batch_size_actor in "${batch_sizes_actor[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",BATCH_SIZE_ACTOR="${batch_size_actor}" launchers/slurm/rover/rover_base.sh
    done
done    
       