#!/bin/bash

seeds="0 1 2 3 4 5 6"

model_path=(
    ""
    ""
)

encoder_path=(
    ""
    ""
)

# Safety check
if [ "${#model_path[@]}" -ne "${#encoder_path[@]}" ]; then
    echo "Error: model_path and encoder_path must have the same length"
    exit 1
fi

for seed in $seeds; do
    for i in "${!model_path[@]}"; do

        sbatch --export=SEED="${seed}",REPLAY_BUFFER_DIR="${model_path[$i]}",ENCODER_PATH="${encoder_path[$i]}" \
            launchers/slurm/offline/ddpg/bowling_score_masked_w_enc_base.sh
    done
done