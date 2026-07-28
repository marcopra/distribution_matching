#!/bin/bash

# Focused operator ablation. Start with one seed; promote winning
# configuration to seeds 1/2/3 after policy deviation becomes non-zero.
seeds="1"

batch_sizes_actor=(
    50000
)

subsamples=(
    5000
)

feature_dims=(
    256
)

kernel_bandwidth_multipliers=(
    0.5
    1.0
)

kernel_types=(
    # gaussian
    inner_product
)

linear_projections=(
    true
    false
)

for seed in $seeds; do
    for batch_size_actor in "${batch_sizes_actor[@]}"; do
        for subsample in "${subsamples[@]}"; do
            for feature_dim in "${feature_dims[@]}"; do
                for kernel_type in "${kernel_types[@]}"; do
                    if [[ "$kernel_type" == "gaussian" ]]; then
                        bandwidths=("${kernel_bandwidth_multipliers[@]}")
                    else
                        # Bandwidth has no meaning for inner-product kernel.
                        # One sentinel prevents duplicate inner-product jobs.
                        bandwidths=("none")
                    fi
                    for kernel_bandwidth_multiplier in "${bandwidths[@]}"; do
                        for linear_projection in "${linear_projections[@]}"; do
                            sbatch \
                                --export=ALL,SEED="$seed",BATCH_SIZE_ACTOR="$batch_size_actor",SUBSAMPLE="$subsample",FEATURE_DIM="$feature_dim",KERNEL_TYPE="$kernel_type",KERNEL_BANDWIDTH_MULTIPLIER="$kernel_bandwidth_multiplier",LINEAR_PROJECTION="$linear_projection" \
                                launchers/slurm/pretraining/atari/montezouma/parallel_rover_nystrom_base.sh
                        done
                    done
                done
            done
        done
    done
done
