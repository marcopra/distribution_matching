#!/bin/bash

seeds="0 1 2"
model_path=(
    "/home/mprattico-iit.local/distribution_matching/data_offline/pong/rover/buffer"
    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",MODEL_PATH="${path}" launchers/slurm/offline_cql/pong_base.sh
    done
done