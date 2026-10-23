#!/bin/bash
set -eo pipefail

# Run from repository root. This script only submits base.sh jobs.
BASE="launchers/slurm/pretraining/pointmaze_subspace/umaze/pixels/baselines/base.sh"
seeds=(1)

agents=(
    rnd_discrete
    icm_apt_discrete
    maxent_discrete
    cic_discrete
    smm_discrete
)
start_xs=(-1.0 1.0 1.0)
start_ys=(1.0 0.0 1.0)

for seed in "${seeds[@]}"; do
    for start_idx in "${!start_xs[@]}"; do
        for agent in "${agents[@]}"; do
            sbatch --export=ALL,SEED="${seed}",AGENT="${agent}",START_X="${start_xs[$start_idx]}",START_Y="${start_ys[$start_idx]}" "${BASE}"
        done
    done
done
