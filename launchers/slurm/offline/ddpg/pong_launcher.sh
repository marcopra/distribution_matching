#!/bin/bash

seeds="3 4"
model_path=(
    "/home/mprattico/distribution_matching/data_offline/pong/1M/random"
    "/home/mprattico/distribution_matching/data_offline/pong/1M/rover"
    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/ddpg/pong_base.sh
    done
done