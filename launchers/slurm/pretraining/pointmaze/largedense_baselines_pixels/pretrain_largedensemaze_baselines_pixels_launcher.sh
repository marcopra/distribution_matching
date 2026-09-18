#!/bin/bash

BASE="launchers/slurm/pretraining/pointmaze/largedense_baselines_pixels/pretrain_largedensemaze_baselines_pixels_base.sh"
seeds=(1)
agents=(rnd_discrete icm_apt_discrete maxent_discrete cic_discrete smm_discrete)

for seed in "${seeds[@]}"; do
    for agent in "${agents[@]}"; do
        sbatch --export=ALL,AGENT="${agent}",SEED="${seed}" "${BASE}"
    done
done
