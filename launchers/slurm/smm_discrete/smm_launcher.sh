#!/bin/bash

seeds="1"
envs=(
# "continuous_four_rooms"
# "continuous_multiple_rooms"
# "multiplerooms10_3x3" 
# "four_rooms5_0" 
# "two_rooms7_0"
"pong"
)
obs_types=(
"pixels"
# "discrete_states"
)


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for obs_type in "${obs_types[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",OBS_TYPE="${obs_type}" launchers/slurm/smm_discrete/smm_base.sh
            
        done
    done
done    
       