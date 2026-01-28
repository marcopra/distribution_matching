#!/bin/bash

seeds="0 1 2"
envs=(
    "multiplerooms5_4x4_0" 
       "multiplerooms5_4x4_1" 
       "multiplerooms5_4x4_2" 
    )
model_path=(
    "none"
    )
obs_type=("discrete_states"
    )

for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            for obs in "${obs_type[@]}"; do
                sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}",OBS_TYPE="${obs}" launchers/slurm/finetuning/ddpg_discrete_baselines/multiplerooms5_4x4_base.sh
            done
        done
    done
done