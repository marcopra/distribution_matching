#!/bin/bash

seeds="0 1 2 3 4 5 6 7 8 9 10" # 3 4 5
model_path=(
    #"/home/mprattico/distribution_matching/data_offline/pong_score_masked/1M/random"
    #"/home/mprattico/distribution_matching/data_offline/pong_score_masked/1M/rover"
    #"/home/mprattico/distribution_matching/data_offline/pong_score_masked/1M/smm"
    #"/home/mprattico/distribution_matching/data_offline/pong_score_masked/1M/cic"
    "/home/mprattico/distribution_matching/data_offline/pong_score_masked/1M/rnd"
    

    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/ddpg/pong_score_masked_base.sh
    done
done
