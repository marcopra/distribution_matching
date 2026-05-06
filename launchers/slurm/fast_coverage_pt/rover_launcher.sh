#!/bin/bash

seeds="0 1 2 3"

envs=(
    "gridworld/maze_108_seed7_env"
    # "gridworld/multiplerooms5_4x4_0"
)


for seed in $seeds; do
    for env in "${envs[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}" "launchers/slurm/fast coverage_pt/rover_base.sh"
    done
done
