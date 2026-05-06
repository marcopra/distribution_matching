#!/bin/bash

seeds="0 1 2 3"

envs=(
    "gridworld/maze_108_seed7_env"
    "gridworld/multiplerooms5_4x4_0"
)

agents=(
    "cic_discrete"
    "maxent_discrete"
    "smm_discrete"
    "icm_apt_discrete"
    "rnd_discrete"
)



for seed in $seeds; do
    for env in "${envs[@]}"; do
        for agent in "${agents[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",AGENT="${agent}" "launchers/slurm/fast_coverage_pt/baseline_base.sh"
        done
    done
done
