#!/bin/bash

seeds="4 5 6"
model_path=(
    # "/home/mprattico/distribution_matching/data_offline/pong/100k/cic"
    "/home/mprattico/distribution_matching/data_offline/pong/800k/random"
    "/home/mprattico/distribution_matching/data_offline/pong/800k/rover_sampling"
    "/home/mprattico/distribution_matching/data_offline/pong/800k/rover"

    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/cql/pong_base.sh
    done
done