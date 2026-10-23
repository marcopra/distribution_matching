#!/bin/bash
set -eo pipefail

# Run from repository root. This script only submits base.sh jobs.
BASE="launchers/slurm/pretraining/pointmaze_subspace/umaze/pixels/rover/base.sh"
seeds=(1)

batch_sizes_actor=(80000)
feature_dims=(1024 2048)
subsample_sizes=(15000)
lambda_regs=(1e-3)
cholesky_tolerances=(1e-5 1e-6)
sink_idxs=(0 1)
kernel_types=(inner_product gaussian)
start_xs=(-1.0 1.0 1.0)
start_ys=(1.0 0.0 1.0)

# Full factorial sweep: 48 jobs. Index 0 matches the saved run's sink schedule.
for seed in "${seeds[@]}"; do
    for start_idx in "${!start_xs[@]}"; do
        for batch_size_actor in "${batch_sizes_actor[@]}"; do
            for feature_dim in "${feature_dims[@]}"; do
                for subsamples in "${subsample_sizes[@]}"; do
                    for lambda_reg in "${lambda_regs[@]}"; do
                        for cholesky_tolerance in "${cholesky_tolerances[@]}"; do
                            for sink_idx in "${sink_idxs[@]}"; do
                                for kernel_type in "${kernel_types[@]}"; do
                                    export_args="ALL,SEED=${seed},START_X=${start_xs[$start_idx]},START_Y=${start_ys[$start_idx]},BATCH_SIZE_ACTOR=${batch_size_actor},FEATURE_DIM=${feature_dim},SUBSAMPLES=${subsamples},LAMBDA_REG=${lambda_reg},CHOLESKY_TOLERANCE=${cholesky_tolerance},SINK_IDX=${sink_idx},DYNAMICS_KERNEL_TYPE=${kernel_type}"
                                    sbatch --export="${export_args}" "${BASE}"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
