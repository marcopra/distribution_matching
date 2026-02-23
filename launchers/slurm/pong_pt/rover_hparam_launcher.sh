#!/bin/bash

seeds="1"
envs=(
    "pong"
)
lr_actors=(
    10
    50
)
# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> linear(0.0, 0.0001, 1_000_000)
#   1 -> linear(0.0, 1, 1_000_000)
#   2 -> linear(0.0, 0.1, 100_000)
sink_idxs=(0 1 2)
batch_sizes_actor=(
    1000
    # 5000
    # 10000
)
feature_dims=(
    128
    512
    1024
)

for seed in $seeds; do
    for env in "${envs[@]}"; do
        for lr_actor in "${lr_actors[@]}"; do
            for sink_idx in "${sink_idxs[@]}"; do
                for batch_size_actor in "${batch_sizes_actor[@]}"; do
                    for feature_dim in "${feature_dims[@]}"; do
                        sbatch --export=SEED="${seed}",ENV="${env}",LR_ACTOR="${lr_actor}",SINK_IDX="${sink_idx}",BATCH_SIZE_ACTOR="${batch_size_actor}",FEATURE_DIM="${feature_dim}" \
                            launchers/slurm/pong_pt/rover_hparam_base.sh
                    done
                done
            done
        done
    done
done
