#!/bin/bash

seeds="1"
envs=(
  gridworld/middle_room
)

feature_dims=(
    "200"
)

# Indices into the sink_schedules array defined in rover_base.sh:
#   0 -> "linear(0.0,0.5,100_000)"
sink_idxs=(0)


discounts=(
    # "0.99"
    "0.9"
)

for seed in $seeds; do
    for env in "${envs[@]}"; do
        for sink_idx in "${sink_idxs[@]}"; do
            for discount in "${discounts[@]}"; do
                for feature in "${feature_dims[@]}"; do
                    sbatch --export=SEED="${seed}",ENV="${env}",SINK_IDX="${sink_idx}",DISCOUNT="${discount}",FEATURE_DIM="${feature}" launchers/slurm/middle_room_pt/rover_base.sh
                done
            done
        done
    done
done
