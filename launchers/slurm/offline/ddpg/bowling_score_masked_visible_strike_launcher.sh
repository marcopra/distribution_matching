#!/bin/bash

seeds="0 1 2 3 4 5 6"
model_path=(
    # "/home/mprattico/distribution_matching/data_offline/bowling_score_masked_visible_strike/1M/random"
    "/home/mprattico/distribution_matching/data_offline/bowling_score_masked_visible_strike/1M/rover"
    # "/home/mprattico/distribution_matching/data_offline/bowling_score_masked_visible_strike/1M/rover_64"
    # "/home/mprattico/distribution_matching/data_offline/bowling_score_masked_visible_strike/1M/cic"
    # "/home/mprattico/distribution_matching/data_offline/bowling_score_masked_visible_strike/1M/rnd"
    # "/home/mprattico/distribution_matching/data_offline/bowling_score_masked_visible_strike/1M/smm"
    # "/home/mprattico/distribution_matching/data_offline/bowling_score_masked_visible_strike/1M/icm_apt"
    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/ddpg/bowling_score_masked_visible_strike_base.sh
    done
done
