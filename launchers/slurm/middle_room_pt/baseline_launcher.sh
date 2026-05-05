#!/bin/bash

seeds="1"

agents=(
    "cic_discrete"
    "maxent_discrete"
    "smm_discrete"
    "icm_apt_discrete"
    "rnd_discrete"
)

for seed in $seeds; do
    for agent in "${agents[@]}"; do
        sbatch --export=SEED="${seed}",AGENT="${agent}" launchers/slurm/middle_room_pt/baseline_base.sh
    done
done
