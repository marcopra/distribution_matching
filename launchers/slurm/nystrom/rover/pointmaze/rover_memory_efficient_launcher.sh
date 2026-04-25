#!/bin/bash

seeds="1"
envs=(
  pointmaze/pointmaze_umaze_goal_1
)


subsamples=(
# "none"
"4000"
"2000"
# "500"
# "100"
# "20"
)

feature_dims=(
    100
    256
    512
)

lr_actors=( 
    10
    100
)

# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> linear(0.0, 0.0001, 1_000_000)
#   1 -> linear(0.0, 0.001,  2_000_000)
#   2 -> linear(0.0, 1,      1_000_000)
#   3 -> linear(0.0, 0.0001, 500_000)
#   4 -> linear(1.0, 1.0,    100_000)

sink_idxs=(0 1) 


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for subsamples in "${subsamples[@]}"; do
            for lr_actor in "${lr_actors[@]}"; do
                for sink_idx in "${sink_idxs[@]}"; do
                    for feature in "${feature_dims[@]}"; do
                        sbatch --export=SEED="${seed}",ENV="${env}",SUBSAMPLES="${subsamples}",LR_ACTOR="${lr_actor}",SINK_IDX="${sink_idx}",FEATURE_DIM="${feature}" launchers/slurm/nystrom/rover/pointmaze/rover_memory_efficient_base.sh
                    done
                done
            done
        done
    done
done    
       