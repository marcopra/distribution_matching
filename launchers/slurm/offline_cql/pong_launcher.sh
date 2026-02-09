#!/bin/bash

seeds="0 1 2"
model_path=(
    # "/home/mprattico/distribution_matching/data_offline/pong/rover/buffer"
    "/home/mprattico-iit.local/distribution_matching/data_offline/pong/cic/buffer"
    "/home/mprattico-iit.local/distribution_matching/data_offline/pong/rnd/buffer"

    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline_cql/pong_base.sh
    done
done