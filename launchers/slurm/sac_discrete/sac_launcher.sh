#!/bin/bash

seeds="0 1 2 3"
model_paths=(
    "/home/mprattico/distribution_matching/models/four_rooms10/smm/smm_1000200.pt"
    "/home/mprattico/distribution_matching/models/four_rooms10/rnd/rnd_500100.pt"
    "/home/mprattico/distribution_matching/models/four_rooms10/icm_apt/icm_apt_500100.pt"
    "/home/mprattico/distribution_matching/models/four_rooms10/dist_matching/policy_operator.npy"
    "/home/mprattico/distribution_matching/models/four_rooms10/dist_matching/uniform_policy_operator.npy"
)
seed_frames="10000 15000 20000"



for seed_frame in $seed_frames; do
    for seed in $seeds; do   
        for model_path in "${model_paths[@]}"; do
            sbatch --export=SEED="${seed}",SEED_FRAMES="${seed_frame}",MODEL_PATH="${model_path}" launchers/slurm/sac_discrete/sac_base.sh
               
        done
    done
done