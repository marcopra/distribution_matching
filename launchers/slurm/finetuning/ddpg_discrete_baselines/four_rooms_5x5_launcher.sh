#!/bin/bash

seeds="0 1"
envs=("four_rooms5_0" 
       "four_rooms5_1" 
       "four_rooms5_2" )
model_path=(
    "none"
    )
obs_type=("pixels"
    )

for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            for obs in "${obs_type[@]}"; do
                sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}",OBS_TYPE="${obs}" launchers/slurm/finetuning/ddpg_discrete_baselines/four_rooms_5x5_base.sh
            done
        done
    done
done