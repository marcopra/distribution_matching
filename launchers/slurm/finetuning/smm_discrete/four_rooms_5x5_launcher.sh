#!/bin/bash

seeds="0 1"
envs=("four_rooms5_0" 
       "four_rooms5_1" 
       "four_rooms5_2" )
model_path=(
    "/home/mprattico/distribution_matching/models/smm/four_rooms_5x5/snapshot.pt"
    )


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}" launchers/slurm/finetuning/smm_discrete/four_rooms_5x5_base.sh
        done
    done
done