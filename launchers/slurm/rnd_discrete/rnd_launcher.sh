#!/bin/bash

seeds="1"
envs=(
"multiplerooms10_3x3" 
# "four_rooms5_0" 
# "two_rooms7_0"
)


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        sbatch --export=SEED="${seed}",ENV="${env}" launchers/slurm/rnd_discrete/rnd_base.sh
            
    done
done