#!/bin/bash
set -eo pipefail

# Run from repository root. This script only submits base.sh jobs.
BASE="launchers/slurm/pretraining/pointmaze_subspace/umaze/states/rover/base.sh"
seeds=(1 2 3)

batch_sizes_actor=(32000)
feature_dims=(256)
subsample_sizes=(5000)
sink_idxs=(1 2)
# "linear(0.0,10.0,50000,100000)"
# "linear(0.0,5.0,50000,250000)"
# "linear(0.0,2.0,50000,250000)"
# "linear(0.0,1.0,50000,250000)"
kernel_types=(gaussian)
start_xs=(1.0 1.0) # -1.0
start_ys=(0.0 1.0) # 1.0

# 6 configurations per seed: start position x sink schedule.
for seed in "${seeds[@]}"; do
    for start_idx in "${!start_xs[@]}"; do
        for batch_size_actor in "${batch_sizes_actor[@]}"; do
            for feature_dim in "${feature_dims[@]}"; do
                for subsamples in "${subsample_sizes[@]}"; do
                    for sink_idx in "${sink_idxs[@]}"; do
                        for kernel_type in "${kernel_types[@]}"; do
                            export_args="ALL,SEED=${seed},START_X=${start_xs[$start_idx]},START_Y=${start_ys[$start_idx]},BATCH_SIZE_ACTOR=${batch_size_actor},FEATURE_DIM=${feature_dim},SUBSAMPLES=${subsamples},SINK_IDX=${sink_idx},DYNAMICS_KERNEL_TYPE=${kernel_type}"
                            sbatch --export="${export_args}" "${BASE}"
                        done
                    done
                done
            done
        done
    done
done
