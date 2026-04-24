#!/bin/bash

seeds="1"

subsamples=(
# "none"
"4000"
"1000"
# "500"
# "100"
# "20"
)

feature_dims=(
    109
)

lr_actors=( 
    10
)
# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> linear(0.0, 0.05, 50_000)
#   1 -> linear(0.0, 0.001, 50_000)
#   2 -> linear(0.0, 1, 50_000)
#   3 -> linear(0.0, 0.0001, 500_000)
#   4 -> linear(1.0, 1.0,    100_000)

sink_idxs=(0 1) 


for seed in $seeds; do   
    for subsamples in "${subsamples[@]}"; do
        for lr_actor in "${lr_actors[@]}"; do
            for sink_idx in "${sink_idxs[@]}"; do
                for feature in "${feature_dims[@]}"; do
                    sbatch --export=SEED="${seed}",SUBSAMPLES="${subsamples}",LR_ACTOR="${lr_actor}",SINK_IDX="${sink_idx}",FEATURE_DIM="${feature}" launchers/slurm/nystrom/rover/subsamples_on_multplerooms/rover_memory_efficient_base.sh
                done
            done
        done
    done
done    
       