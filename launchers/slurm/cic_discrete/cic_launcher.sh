#!/bin/bash

seeds="0" #1 2"
envs=(
# "multiplerooms10_3x3" 
# "four_rooms10_2" 
# "two_rooms15_0"
# "pong"
"pong_score_masked"
"tennis_score_masked"
# "tennis"
"bowling_score_masked"
# "mariobros_score_masked"
)

obs_types=(
"pixels"
# "discrete_states"
)


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for obs_type in "${obs_types[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",OBS_TYPE="${obs_type}" launchers/slurm/cic_discrete/cic_base.sh
        done
    done
done