#!/bin/bash

# One seed by default, matching saved Rover run. Add seeds here for paper repeats.
seeds="1"

envs=(
    "gridworld/maze_108_seed7_env"
    "gridworld/maze_1000_env"
    # "gridworld/maze_200_seed7_env"
    # "gridworld/maze_500_seed7_env"
)

agents=(
    "cic_discrete"
    "rnd_discrete"
    "smm_discrete"
    "icm_apt_discrete"
)

base_script="launchers/slurm/pretraining/gridworld/maze/baseline_base.sh"

for seed in $seeds; do
    for env in "${envs[@]}"; do
        for agent in "${agents[@]}"; do
            sbatch \
                --export=ALL,SEED="${seed}",ENV="${env}",AGENT="${agent}" \
                "${base_script}"
        done
    done
done
