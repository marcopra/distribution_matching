#!/bin/bash

seeds="0 1 2 3 4 5 6"
replay_buffer_dirs=(
    "data_offline/bowling_score_masked/1M/icm_apt50"
)

encoder_paths=(
    "data_offline/bowling_score_masked/1M/icm_apt50/models/pixels/gym/icm_apt/0/snapshot.pt"
)

feature_dims=(
    "50"
    )

if [ "${#replay_buffer_dirs[@]}" -ne "${#encoder_paths[@]}" ]; then
    echo "Error: replay_buffer_dirs and encoder_paths must have the same length."
    exit 1
fi

#check feature dims lenght
if [ "${#replay_buffer_dirs[@]}" -ne "${#feature_dims[@]}" ]; then
    echo "Error: replay_buffer_dirs and feature dims must have the same length."
    exit 1
fi


for seed in $seeds; do
    for i in "${!replay_buffer_dirs[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${replay_buffer_dirs[$i]}",ENCODER_PATH="${encoder_paths[$i]}",FEATURE_DIM="${feature_dims[$i]}" launchers/slurm/rebuttals/ddpg_with_learned_encoder/rgb_bowling_score_masked_base.sh
    done
done
