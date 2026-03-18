#!/bin/bash

seeds="0 1 2 3 4 5 6"
model_path=(
    "/home/mprattico/distribution_matching/data_offline/pong_score_masked/1M/rover"
    

    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/ddpg_with_learned_encoder/pong_score_masked_base.sh
    done
done
