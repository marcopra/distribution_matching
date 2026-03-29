#!/bin/bash

seeds="1"
envs=(
  montezoumarevenge_score_masked
)
lr_actors=(
    # 10 
    # 100
    200
)
# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> linear(0.0, 0.0001, 1_000_000)
#   1 -> linear(0.0, 0.001,  2_000_000)
#   2 -> linear(0.0, 1,      1_000_000)
#   3 -> linear(0.0, 0.0001, 500_000)
#   4 -> linear(1.0, 1.0,    100_000)

sink_idxs=(0 1) 
batch_sizes_actor=(
    100000
    # 30000
    # 10000
)
susbamples=(
   "5000"
   "10000"
)

for seed in $seeds; do
    for env in "${envs[@]}"; do
        for lr_actor in "${lr_actors[@]}"; do
            for sink_idx in "${sink_idxs[@]}"; do
                for batch_size_actor in "${batch_sizes_actor[@]}"; do
                    for sub in "${susbamples[@]}"; do
                        sbatch --export=SEED="${seed}",ENV="${env}",LR_ACTOR="${lr_actor}",SINK_IDX="${sink_idx}",BATCH_SIZE_ACTOR="${batch_size_actor}",SUBSAMPLES="${sub}" launchers/slurm/rebuttals/rover_montezouma/rover_base.sh
                    done
                done
            done
        done
    done
done
