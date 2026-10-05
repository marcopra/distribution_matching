#!/bin/bash

BASE="launchers/slurm/pretraining/pointmaze/umaze_rover_nystrom_subspace/base.sh"

seeds=(1)

nystrom_cholesky_tolerances=(0 1e-6)

kernel_bandwidth_mults=(1.0)

batch_sizes_actor=(16000 32000)

nystrom_points=(8000 10000)

pca_truncations=(5000)

# Indices into sink_schedules in base.sh:
#   1 -> 0.0
#   2 -> linear(0.0,0.8,25000,50000)
sink_idxs=(1 2)

for seed in "${seeds[@]}"; do
    for cholesky_tolerance in "${nystrom_cholesky_tolerances[@]}"; do
        for bandwidth_mult in "${kernel_bandwidth_mults[@]}"; do
            for batch_size_actor in "${batch_sizes_actor[@]}"; do
                for subsample in "${nystrom_points[@]}"; do
                    for pca_truncation in "${pca_truncations[@]}"; do
                        for sink_idx in "${sink_idxs[@]}"; do
                            sbatch \
                                --export=ALL,SEED="${seed}",CHOLESKY_TOLERANCE="${cholesky_tolerance}",KERNEL_BANDWIDTH_MULT="${bandwidth_mult}",BATCH_SIZE_ACTOR="${batch_size_actor}",SUBSAMPLE="${subsample}",PCA_TRUNCATION="${pca_truncation}",SINK_IDX="${sink_idx}" \
                                "${BASE}"
                        done
                    done
                done
            done
        done
    done
done
