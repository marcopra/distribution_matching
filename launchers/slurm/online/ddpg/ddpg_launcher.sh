#!/bin/bash

seeds="0 1 2 3 4 5 6"
envs=(
    # "bowling_score_masked_visible_strike"
    # "bowling_score_masked"
    # "pong_score_masked"
    # "tennis_score_masked"
    "mariobros_score_masked"
    )


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        sbatch --export=SEED="${seed}",ENV="${env}" launchers/slurm/online/ddpg/ddpg_base.sh
    done
done
