#!/bin/bash

BASE="launchers/slurm/pretraining/pointmaze/largedense_rover_nystrom/base.sh"

seeds=(1)

# Indices into kernel_bandwidth_schedules in base.sh:
#   0 -> linear(0.05, 0.3,  500000)
#   1 -> linear(0.1,  0.25, 500000)
#   2 -> linear(0.05,  0.2, 500000)
#   3 -> linear(0.1,  0.2, 500000)
#   4 -> linear(0.05,  0.15, 500000)
#   5 -> linear(0.01,  0.18, 500000)
kernel_bandwidth_idxs=(4 5)

nystrom_points=(
    # 4000
    # 8000
    12000
)

batch_sizes_actor=(
    32000
    #48000
)

# Indices into sink_schedules in base.sh:
#   0 -> linear(0.0, 0.001, 500000)
#   1 -> linear(0.0, 0.01,  500000)
#   2 -> linear(0.0, 0.1,   500000)
#   3 -> linear(0.0, 0.8,   500000)
#   4 -> 0.8
sink_idxs=(3)

for seed in "${seeds[@]}"; do
    for bandwidth_idx in "${kernel_bandwidth_idxs[@]}"; do
        for subsample in "${nystrom_points[@]}"; do
            for batch_size_actor in "${batch_sizes_actor[@]}"; do
                for sink_idx in "${sink_idxs[@]}"; do
                    sbatch \
                        --export=ALL,SEED="${seed}",KERNEL_BANDWIDTH_IDX="${bandwidth_idx}",SUBSAMPLE="${subsample}",BATCH_SIZE_ACTOR="${batch_size_actor}",SINK_IDX="${sink_idx}" \
                        "${BASE}"
                done
            done
        done
    done
done
