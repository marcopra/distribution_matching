#!/bin/bash

seeds=" 7 8 9"
model_path=(
    "data_offline/tennis_score_masked/1M/random"
    "data_offline/tennis_score_masked/1M/rover"
    "data_offline/tennis_score_masked/1M/cic"
    "data_offline/tennis_score_masked/1M/rnd"
    "data_offline/tennis_score_masked/1M/smm"

    )


for seed in $seeds; do   
    for path in "${model_path[@]}"; do
        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${path}" launchers/slurm/offline/cql/tennis_score_masked_base.sh
    done
done
