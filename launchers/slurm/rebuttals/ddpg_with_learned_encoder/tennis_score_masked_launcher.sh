#!/bin/bash

seeds="0 1 2 3 4 5 6"
replay_buffer_dirs=(
    "/home/mprattico/distribution_matching/data_offline/tennis_score_masked/grayscale/1M/rover"
)

encoder_paths=(
    "/home/mprattico/distribution_matching/data_offline/tennis_score_masked/grayscale/1M/rover/models/pixels/gym/dist_matching/1/snapshot.pt"
)

if [ "${#replay_buffer_dirs[@]}" -ne "${#encoder_paths[@]}" ]; then
    echo "Error: replay_buffer_dirs and encoder_paths must have the same length."
    exit 1
fi


for seed in $seeds; do
    for i in "${!replay_buffer_dirs[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${replay_buffer_dirs[$i]}",ENCODER_PATH="${encoder_paths[$i]}" launchers/slurm/rebuttals/ddpg_with_learned_encoder/tennis_score_masked_base.sh
    done
done
