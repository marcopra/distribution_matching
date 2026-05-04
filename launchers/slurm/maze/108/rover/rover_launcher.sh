#!/bin/bash

seeds="0 1 2 3 4 5 6"

envs=(
    "gridworld/maze_108_seed7_env_3"
)

actor_lrs=(
    "1e-4"
    "1e-7"
)

obs_types=(
    "pixels"
)

agents=(
    "ddpg_discrete_with_kernel_actor"
)

model_paths=(
    "/home/mprattico/distribution_matching/models/maze/108/pixels/rover/snapshot.pt"
)

feature_dims=(
    "200"
)

dataset_dims=(
    "4096"
)

for seed in $seeds; do
    for env in "${envs[@]}"; do
        for agent in "${agents[@]}"; do
            for path in "${model_paths[@]}"; do
                for lr in "${actor_lrs[@]}"; do
                    for feature in "${feature_dims[@]}"; do
                        for dataset in "${dataset_dims[@]}"; do
                            for obs in "${obs_types[@]}"; do
                                sbatch --export=SEED="${seed}",ENV="${env}",AGENT="${agent}",MODEL_PATH="${path}",ACTOR_LR="${lr}",FEATURE_DIM="${feature}",DATASET_DIM="${dataset}",OBS_TYPE="${obs}" launchers/slurm/maze/108/rover/rover_base.sh
                            done
                        done
                    done
                done
            done
        done
    done
done
