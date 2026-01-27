#!/bin/bash

seeds="0 1 2"
envs=("two_rooms7_0" 
    #    "two_rooms7_1" 
       "two_rooms7_2" )
model_path=(
    "none"
    )
obs_type=("discrete_states"
    )

for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for path in "${model_path[@]}"; do
            for obs in "${obs_type[@]}"; do
                sbatch --export=SEED="${seed}",ENV="${env}",MODEL_PATH="${path}",OBS_TYPE="${obs}" launchers/slurm/finetuning/ddpg_discrete_baselines/two_rooms_7x7_base.sh
            done
        done
    done
done