#!/bin/bash
set -eo pipefail

# Run from repository root. This script only submits base.sh jobs.
BASE="launchers/slurm/pretraining/pointmaze_subspace/largedense/states/rover/base.sh"
seeds=(1)

batch_sizes_actor=(20000 32000)
feature_dims=(64 256)
subsample_sizes=(5000 10000)
sink_idxs=(0 1)
kernel_types=(gaussian)

# 16 configurations per seed: schedule x batch x feature dim x subsample size.
for seed in "${seeds[@]}"; do
    for batch_size_actor in "${batch_sizes_actor[@]}"; do
        for feature_dim in "${feature_dims[@]}"; do
            for subsamples in "${subsample_sizes[@]}"; do
                for sink_idx in "${sink_idxs[@]}"; do
                    for kernel_type in "${kernel_types[@]}"; do
                        export_args="ALL,SEED=${seed},BATCH_SIZE_ACTOR=${batch_size_actor},FEATURE_DIM=${feature_dim},SUBSAMPLES=${subsamples},SINK_IDX=${sink_idx},DYNAMICS_KERNEL_TYPE=${kernel_type}"
                        sbatch --export="${export_args}" "${BASE}"
                    done
                done
            done
        done
    done
done
