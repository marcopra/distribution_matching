#!/bin/bash

seeds="0 1 2 3 4"
model_paths=(
    ""
    ""
)
seed_frames="5000 10000 15000"



for seed_frame in $seed_frames; do
    for seed in $seeds; do   
        for model_path in "${model_paths[@]}"; do
            sbatch --export=SEED="${seed}",SEED_FRAMES="${seed_frame}",MODEL_PATH="${model_path}" launchers/slurm/sac_discrete/sac_base.sh
               
        done
    done
done