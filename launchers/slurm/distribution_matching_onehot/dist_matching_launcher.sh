#!/bin/bash





for seed_frame in $seed_frames; do
    for seed in $seeds; do   
        for model_path in "${model_paths[@]}"; do
            sbatch --export=VAR="${}",VAR2="${seed_frame}",MODEL_PATH="${model_path}" launchers/slurm/sac_discrete/sac_base.sh
               
        done
    done
done