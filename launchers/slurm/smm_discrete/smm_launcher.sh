#!/bin/bash

seeds="0 1 2"
envs="multiplerooms10_3x3 four_rooms10_2 two_rooms15_0"



for seed in $seeds; do   
    for env in "${envs[@]}"; do
        sbatch --export=SEED="${seed}",ENV="${env}" launchers/slurm/smm_discrete/smm_base.sh
            
    done
done