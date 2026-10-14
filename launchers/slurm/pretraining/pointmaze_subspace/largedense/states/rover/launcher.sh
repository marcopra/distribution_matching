#!/bin/bash
set -eo pipefail

# Run from repository root. This script only submits base.sh jobs.
BASE="launchers/slurm/pretraining/pointmaze_subspace/largedense/states/rover/base.sh"
seeds=(1)

batch_sizes_actor=(16000 32000)
feature_dims=(256 512)
subsample_sizes=(4000 8000)
sink_idxs=(0 1)

# Full factorial sweep: 2^4 = 16 jobs per seed.
for seed in "${seeds[@]}"; do
    for batch_size_actor in "${batch_sizes_actor[@]}"; do
        for feature_dim in "${feature_dims[@]}"; do
            for subsamples in "${subsample_sizes[@]}"; do
                for sink_idx in "${sink_idxs[@]}"; do
                    export_args="ALL,SEED=${seed},BATCH_SIZE_ACTOR=${batch_size_actor},FEATURE_DIM=${feature_dim},SUBSAMPLES=${subsamples},SINK_IDX=${sink_idx}"
                    sbatch --export="${export_args}" "${BASE}"
                done
            done
        done
    done
done
