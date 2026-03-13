#!/bin/bash

seeds="10 11 12 13"
envs=(
    "bowling_score_masked_visible_strike"
    # "bowling_score_masked"
    "pong_score_masked"
    "tennis_score_masked"
    )


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        sbatch --export=SEED="${seed}",ENV="${env}" launchers/slurm/online/ddpg/ddpg_base.sh
    done
done
