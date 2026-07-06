#!/bin/bash

seeds="0"

envs=(
    "gridworld/maze_108_seed7_env"
    # "gridworld/multiplerooms5_4x4_0"
)

obs_types=(
    "pixels"
    "discrete_states"
)

for seed in $seeds; do
    for env in "${envs[@]}"; do
        for obs_type in "${obs_types[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",OBS_TYPE="${obs_type}" "launchers/slurm/pretraining/gridworld/fast_coverage/rover_base.sh"
        done
    done
done
