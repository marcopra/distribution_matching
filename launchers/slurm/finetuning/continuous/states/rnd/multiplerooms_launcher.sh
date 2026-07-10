#!/bin/bash

seeds="0 1 2"
envs=("continuous_multiple_rooms_0" 
       "continuous_multiple_rooms_1" 
       "continuous_multiple_rooms_2" 
    )
model_path=(
    "models/continuous/rnd/multiplerooms/states/snapshot.pt"
    )


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}" launchers/slurm/finetuning/continuous/states/rnd/multiplerooms_base.sh
        done
    done
done