#!/bin/bash

seeds="0 1 2"
envs=("continuous_four_rooms_0" 
       "continuous_four_rooms_1" 
       "continuous_four_rooms_2" )
model_path=(
    "none"
    )


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}" launchers/slurm/finetuning/continuous/states/ddpg/four_rooms_base.sh
        done
    done
done