#!/bin/bash

seeds="0 1 2"
envs=("multiplerooms10_3x3_0" 
       "multiplerooms10_3x3_1" 
       "multiplerooms10_3x3_2" 
    )
model_path=(
    "none"
    )
obs_type=("pixels"
    )

for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            for obs in "${obs_type[@]}"; do
                sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}",OBS_TYPE="${obs}" launchers/slurm/finetuning/ddpg_discrete_baselines/multiplerooms10_3x3_base.sh
            done
        done
    done
done