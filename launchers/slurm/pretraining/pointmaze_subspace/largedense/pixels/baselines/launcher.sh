#!/bin/bash
set -eo pipefail

# Run from repository root. This script only submits base.sh jobs.
BASE="launchers/slurm/pretraining/pointmaze_subspace/largedense/pixels/baselines/base.sh"
seeds=(1)

agents=(
    rnd_discrete
    icm_apt_discrete
    maxent_discrete
    cic_discrete
    smm_discrete
)

for seed in "${seeds[@]}"; do
    for agent in "${agents[@]}"; do
        sbatch --export=ALL,SEED="${seed}",AGENT="${agent}" "${BASE}"
    done
done
