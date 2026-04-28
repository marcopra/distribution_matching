#!/bin/bash
seeds="1"
envs=(
  atari/pong_score_masked
#   atari/bowling_score_masked_visible_strike
)


subsamples=(
# "none"
# "8000"
"4000"
# "500"
# "100"
# "20"
)

feature_dims=(
    64
    # 256
    512
)

lr_actors=( 
    1000
)

# Indices into the sink_schedules array defined in rover_hparam_base.sh:
#   0 -> linear(0.0, 0.0001, 1_000_000)
#   1 -> linear(0.0, 0.001,  2_000_000)
#   2 -> linear(0.0, 1,      1_000_000)
#   3 -> linear(0.0, 0.0001, 500_000  )
#   4 -> linear(1.0, 1.0,    100_000  )

sink_idxs=(0 1) # 2 4) 


for seed in $seeds; do   
    for env in "${envs[@]}"; do
        for subsamples in "${subsamples[@]}"; do
            for lr_actor in "${lr_actors[@]}"; do
                for sink_idx in "${sink_idxs[@]}"; do
                    for feature in "${feature_dims[@]}"; do
                        sbatch --export=SEED="${seed}",ENV="${env}",SUBSAMPLES="${subsamples}",LR_ACTOR="${lr_actor}",SINK_IDX="${sink_idx}",FEATURE_DIM="${feature}" launchers/slurm/nystrom/rover/atari/rover_memory_efficient_base.sh
                    done
                done
            done
        done
    done
done    
       