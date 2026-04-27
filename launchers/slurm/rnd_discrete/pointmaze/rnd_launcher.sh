#!/bin/bash

seeds="0"
envs=(
pointmaze_umaze_goal_1
)
obs_types=(
"pixels"
# "discrete_states"
)

feature_dims=(
    # "50"
    "64"
    )

for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for obs_type in "${obs_types[@]}"; do
            for feature_dim in "${feature_dims[@]}"; do
                sbatch --export=SEED="${seed}",ENV="${env}",OBS_TYPE="${obs_type}",FEATURE_DIM="${feature_dim}"  launchers/slurm/rnd_discrete/pointmaze/rnd_base.sh
            done
        done
    done
done