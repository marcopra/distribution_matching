#!/bin/bash

seeds="0"

envs=(
pointmaze/pointmaze_umaze_goal_1
)


feature_dims=(
    # "50"
    "64"
    )

for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for feature_dim in "${feature_dims[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",FEATURE_DIM="${feature_dim}"  launchers/slurm/rnd_discrete/pointmaze/rnd_base.sh
        done
    done
done