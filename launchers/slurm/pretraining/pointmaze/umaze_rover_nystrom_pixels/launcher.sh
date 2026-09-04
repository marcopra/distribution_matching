#!/bin/bash

BASE="launchers/slurm/pretraining/pointmaze/umaze_rover_nystrom_pixels/base.sh"

seeds=(1)

# Indices into kernel_bandwidth_schedules in base.sh:
# 0 -> "0.28"
# 1 -> "0.2"
# 2 -> "0.1"
# 3 -> "0.15"
kernel_bandwidth_idxs=(0 1 2 3)

feature_dims=(64 128)

nystrom_points=(4000)

batch_sizes_actor=(16000)

# Indices into sink_schedules in base.sh:
#   0 -> linear(0.0, 0.001, 500000)
#   1 -> linear(0.0, 0.01,  500000)
#   2 -> linear(0.0, 0.1,   500000)
#   3 -> linear(0.0, 0.8,   500000)
#   4 -> 0.8
sink_idxs=(2)

for seed in "${seeds[@]}"; do
    for bandwidth_idx in "${kernel_bandwidth_idxs[@]}"; do
        for feature_dim in "${feature_dims[@]}"; do
            for subsample in "${nystrom_points[@]}"; do
                for batch_size_actor in "${batch_sizes_actor[@]}"; do
                    for sink_idx in "${sink_idxs[@]}"; do
                        sbatch \
                            --export=ALL,SEED="${seed}",KERNEL_BANDWIDTH_IDX="${bandwidth_idx}",FEATURE_DIM="${feature_dim}",SUBSAMPLE="${subsample}",BATCH_SIZE_ACTOR="${batch_size_actor}",SINK_IDX="${sink_idx}" \
                            "${BASE}"
                    done
                done
            done
        done
    done
done
