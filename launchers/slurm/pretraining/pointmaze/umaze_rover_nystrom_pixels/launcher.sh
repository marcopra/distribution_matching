#!/bin/bash

BASE="launchers/slurm/pretraining/pointmaze/umaze_rover_nystrom_pixels/base.sh"

seeds=(1)

feature_dims=(512)

feature_modes=(l1)

lambda_regs=(1e-3) # 1e-4 1e-5 1e-7)

nystrom_points=(15000 20000)
pca_truncations=(5000 10000)

batch_sizes_actor=(32000)

# Indices into sink_schedules in base.sh: 
# 1 -> 0.0; 
# 2 -> linear(0.0,1.0,25000,50000)
sink_idxs=(1 2)

kernels=(inner_product gaussian)
encoder_modes=(scratch pretrained_finetune pretrained_frozen)
cholesky_tolerances=(0 1e-6)
kernel_bandwidth_mult=1.0

for seed in "${seeds[@]}"; do
    for feature_dim in "${feature_dims[@]}"; do
        for feature_mode in "${feature_modes[@]}"; do
            for lambda_reg in "${lambda_regs[@]}"; do
                for subsample in "${nystrom_points[@]}"; do
                    for pca_truncation in "${pca_truncations[@]}"; do
                        for batch_size_actor in "${batch_sizes_actor[@]}"; do
                            for sink_idx in "${sink_idxs[@]}"; do
                                for kernel_type in "${kernels[@]}"; do
                                    for encoder_mode in "${encoder_modes[@]}"; do
                                        if [[ "${kernel_type}" == "inner_product" ]]; then
                                            selected_tolerances=("${cholesky_tolerances[@]}")
                                        else
                                            selected_tolerances=(1e-5)
                                        fi
                                        for cholesky_tolerance in "${selected_tolerances[@]}"; do
                                            sbatch \
                                                --export=ALL,SEED="${seed}",FEATURE_DIM="${feature_dim}",FEATURE_MODE="${feature_mode}",LAMBDA_REG="${lambda_reg}",SUBSAMPLE="${subsample}",PCA_TRUNCATION="${pca_truncation}",BATCH_SIZE_ACTOR="${batch_size_actor}",SINK_IDX="${sink_idx}",KERNEL_TYPE="${kernel_type}",KERNEL_BANDWIDTH_MULT="${kernel_bandwidth_mult}",ENCODER_MODE="${encoder_mode}",CHOLESKY_TOLERANCE="${cholesky_tolerance}" \
                                                "${BASE}"
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
