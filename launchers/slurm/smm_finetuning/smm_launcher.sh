#!/bin/bash

seeds="0 1 2 3"
model_paths=(
    "/home/mprattico/distribution_matching/models/four_rooms10/smm/smm_1000200.pt"
)
seed_frames="10000 15000 20000"



for seed_frame in $seed_frames; do
    for seed in $seeds; do   
        for model_path in "${model_paths[@]}"; do
            sbatch --export=SEED="${seed}",SEED_FRAMES="${seed_frame}",MODEL_PATH="${model_path}" launchers/slurm/smm_finetuning/smm_base.sh
               
        done
    done
done