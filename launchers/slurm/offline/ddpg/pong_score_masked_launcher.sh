#!/bin/bash

seeds="4 5"
model_path=(
    "/home/mprattico/distribution_matching/data_offline/pong_score_masked/1M/random"
    "/home/mprattico/distribution_matching/data_offline/pong_score_masked/1M/rover"
    # "/home/mprattico/distribution_matching/data_offline/pong_score_masked/100k/random"
    # "/home/mprattico/distribution_matching/data_offline/pong_score_masked/100k/rover"
    # "/home/mprattico/distribution_matching/data_offline/pong_score_masked/500k/random"
    # "/home/mprattico/distribution_matching/data_offline/pong_score_masked/500k/rover"
    # "/home/mprattico/distribution_matching/data_offline/pong/100k/random"
    # "/home/mprattico/distribution_matching/data_offline/pong/100k/cic"
    # "/home/mprattico/distribution_matching/data_offline/pong/100k/rover_sampling"

    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/ddpg/pong_score_masked_base.sh
    done
done