#!/bin/bash

seeds="0"

agents=(
    rnd_discrete
    icm_apt_discrete
    maxent_discrete
    cic_discrete
    smm_discrete
)

for seed in $seeds; do
    for agent in "${agents[@]}"; do
        sbatch --export=AGENT="${agent}",SEED="${seed}" \
            launchers/slurm/umaze_baselines/pretrain_umaze_baselines_base.sh
    done
done
