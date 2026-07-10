#!/bin/bash

seeds="1"
envs=(
"gridworld/multiplerooms5_4x4_0"
)
batch_sizes_actor=(
5000
)


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for batch_size_actor in "${batch_sizes_actor[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",BATCH_SIZE_ACTOR="${batch_size_actor}" launchers/slurm/pretraining/rover/multiplerooms/rover_base.sh
        done
    done
done    
       