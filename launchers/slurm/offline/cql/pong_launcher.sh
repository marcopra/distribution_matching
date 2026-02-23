#!/bin/bash

seeds="0 1 2"
model_path=(
    # "/home/mprattico/distribution_matching/data_offline/pong/100k/cic"
    # "/home/mprattico/distribution_matching/data_offline/pong/100k/random"
    "/home/mprattico/distribution_matching/data_offline/pong/100k/rover"

    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/cql/pong_base.sh
    done
done