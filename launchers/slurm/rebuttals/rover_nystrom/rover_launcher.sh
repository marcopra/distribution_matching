#!/bin/bash

seeds="1"
envs=(
  
)
lr_actors=(
    # 10 
    # 100
    200
)
# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> linear(0.0, 0.0, 50000)
#   1 -> linear(0.0, 0.005,  50000)
#   2 -> linear(0.0, 1,      50000)
#   3 -> linear(0.1, 0.1,    50_000)

sink_idxs=(0 1 2 3) #
batch_sizes_actor=(
    30000
)
susbamples=(
   "null"
   "100"
   "1000"
   "5000"
   "10000"
)

for seed in $seeds; do
    for env in "${envs[@]}"; do
        for lr_actor in "${lr_actors[@]}"; do
            for sink_idx in "${sink_idxs[@]}"; do
                for batch_size_actor in "${batch_sizes_actor[@]}"; do
                    for sub in "${susbamples[@]}"; do
                        sbatch --export=SEED="${seed}",ENV="${env}",LR_ACTOR="${lr_actor}",SINK_IDX="${sink_idx}",BATCH_SIZE_ACTOR="${batch_size_actor}",SUBSAMPLES="${sub}" \
                            launchers/slurm/rebuttals/rover_nystrom/rover_base.sh
                    done
                done
            done
        done
    done
done
