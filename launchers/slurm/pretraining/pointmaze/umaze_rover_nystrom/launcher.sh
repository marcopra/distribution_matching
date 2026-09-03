#!/bin/bash

BASE="launchers/slurm/pretraining/pointmaze/umaze_rover_nystrom/base.sh"

seeds=(1)

kernel_bandwidth=(
    # 0.01
    # 0.05
    0.1
    0.2
    0.3
    # 0.7
    # 1.0
    # 2.0

)

nystrom_points=(
    4000
    # 8000
    # 16000
)

batch_sizes_actor=(
    16000
    # 32000
    # 48000
)

# Indices into sink_schedules in base.sh:
#   0 -> linear(0.0, 0.001, 500000)
#   1 -> linear(0.0, 0.01,  500000)
#   2 -> linear(0.0, 0.1,   500000)
#   3 -> linear(0.0, 1.0,   500000)
#   4 -> 0.0
sink_idxs=(3) 

for seed in "${seeds[@]}"; do
    for bandwidth_mult in "${kernel_bandwidth[@]}"; do
        for subsample in "${nystrom_points[@]}"; do
            for batch_size_actor in "${batch_sizes_actor[@]}"; do
                for sink_idx in "${sink_idxs[@]}"; do
                    sbatch \
                        --export=ALL,SEED="${seed}",KERNEL_BANDWIDTH="${bandwidth_mult}",SUBSAMPLE="${subsample}",BATCH_SIZE_ACTOR="${batch_size_actor}",SINK_IDX="${sink_idx}" \
                        "${BASE}"
                done
            done
        done
    done
done
