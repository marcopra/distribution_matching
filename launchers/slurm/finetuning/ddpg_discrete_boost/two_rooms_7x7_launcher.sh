#!/bin/bash

seeds="0 1 2"
envs=("two_rooms7_0" 
    #    "two_rooms7_1" 
       "two_rooms7_2" )
model_path=(
  "/home/mprattico/distribution_matching/models/boost/two_rooms_7x7/snapshot.pt"
  )


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}" launchers/slurm/finetuning/ddpg_discrete_boost/two_rooms_7x7_base.sh
        done
    done
done