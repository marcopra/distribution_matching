#!/bin/bash

seeds="0 1 2 3 4 5 6" # 3 4 5
model_path=(
    #"data_offline/pong_score_masked/1M/random"
    #"data_offline/pong_score_masked/1M/rover"
    "data_offline/pong_score_masked/1M/smm"
    #"data_offline/pong_score_masked/1M/cic"
    # "data_offline/pong_score_masked/1M/rnd"
    "data_offline/pong_score_masked/1M/icm_apt50"
    

    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/ddpg/pong_score_masked_base.sh
    done
done
