#!/bin/bash

seeds="7 8 9 10"
model_path=(
    "data_offline/bowling_score_masked/1M/grayscale/cic"
    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/ddpg/grayscale_bowling_score_masked_base.sh
    done
done
