#!/bin/bash

seeds="0 1 2"
envs=("multiplerooms5_4x4_0" 
       "multiplerooms5_4x4_1" 
       "multiplerooms5_4x4_2")
model_path=(
  "/home/mprattico/distribution_matching/models/boost/multiplerooms5_4x4/snapshot.pt"
  )


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}" launchers/slurm/finetuning/ddpg_boost/multiplerooms5_4x4_base.sh
        done
    done
done