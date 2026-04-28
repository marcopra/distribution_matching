#!/bin/bash

seeds="1"
envs=(
  pointmaze/pointmaze_umaze_goal_1
)


batch_size_actors=(
# "none"
"8200"
"4100"
# "2000"
# "500"
# "100"
# "20"
)

feature_dims=(
    # 64
    # 128
    256
    # 512
)

lr_actors=( 
    10
)

# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> linear(0.0, 0.5,    500_000)
#   1 -> linear(0.0, 0.01,   500_000)
#   2 -> linear(0.0, 1,      500_000)
#   3 -> linear(0.0, 0.0001, 500_000)
#   4 -> linear(1.0, 1.0,    100_000)

sink_idxs=(0 1) 


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for bs in "${batch_size_actors[@]}"; do
            for lr_actor in "${lr_actors[@]}"; do
                for sink_idx in "${sink_idxs[@]}"; do
                    for feature in "${feature_dims[@]}"; do
                        sbatch --export=SEED="${seed}",ENV="${env}",BATCH_SIZE_ACTOR="${bs}",LR_ACTOR="${lr_actor}",SINK_IDX="${sink_idx}",FEATURE_DIM="${feature}" launchers/slurm/rover/pointmaze/rover_base.sh
                    done
                done
            done
        done
    done
done    
       