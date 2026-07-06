#!/bin/bash

seeds="1"
envs=(
# "continuous_four_rooms"
# "continuous_multiple_rooms"
# "multiplerooms10_3x3" 
# "four_rooms5_0" 
# "two_rooms7_0"
# "pong"
# "pong_score_masked"
# "tennis_score_masked"
"bowling_score_masked"
# "mariobros_score_masked"
# "montezoumarevenge_score_masked"
)
obs_types=(
"pixels"
# "discrete_states"
)

feature_dims=(
    # "50"
    "512"
    )

for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for obs_type in "${obs_types[@]}"; do
            for feature_dim in "${feature_dims[@]}"; do
                sbatch --export=SEED="${seed}",ENV="${env}",OBS_TYPE="${obs_type}",FEATURE_DIM="${feature_dim}" launchers/slurm/pretraining/baselines/discrete/smm/smm_base.sh
            done
        done
    done
done    
       