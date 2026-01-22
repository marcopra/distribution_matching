#!/bin/bash

seeds="0 1 2"
envs=("multiplerooms10_3x3_0" 
       "multiplerooms10_3x3_1" 
       "multiplerooms10_3x3_2" 
    )
model_path=(
    "/home/mprattico/distribution_matching/models/smm/multiplerooms10_3x3/snapshot.pt"
    "/home/mprattico/distribution_matching/models/rnd/multiplerooms10_3x3/snapshot.pt"
    "none"
    )


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}" launchers/slurm/finetuning/ddpg_discrete_baselines/multiplerooms10_3x3_base.sh
        done
    done
done