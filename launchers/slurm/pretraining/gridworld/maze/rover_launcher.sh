#!/bin/bash

seeds="1"
envs=(
  gridworld/maze_500_seed7_env
#   gridworld/maze_200_seed7_env
)


batch_sizes=(

"10000"
# "5000"


)

feature_dims=(
    # 210
    600
)



# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> "linear(0.0, 0.05,   500_000)"
#   1 -> "linear(0.0, 0.01,   500_000)"
#   2 -> "linear(0.0, 0.001,  500_000)"
#   3 -> "linear(0.0, 0.1,    990_000)"


sink_idxs=(1 2 3) # 2 4) 


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for sink_idx in "${sink_idxs[@]}"; do
                for feature in "${feature_dims[@]}"; do
                    sbatch --export=SEED="${seed}",ENV="${env}",BATCH_SIZE_ACTOR="${batch_size}",SINK_IDX="${sink_idx}",FEATURE_DIM="${feature}" launchers/slurm/pretraining/gridworld/maze/rover_base.sh
                done
            done
        done
    done
done    
       