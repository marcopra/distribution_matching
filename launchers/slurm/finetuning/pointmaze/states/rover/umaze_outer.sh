#!/bin/bash
set -euo pipefail

INNER="launchers/slurm/finetuning/pointmaze/states/rover/umaze_inner.sh"
MODEL_PATH="/home/mprattico-iit.local/distribution_matching/models/pointmaze/umaze/states/rover/models/states/gym/dist_matching/1/snapshot.pt"
ENVS=(
    pointmaze/pointmaze_umaze_goal_1
    pointmaze/pointmaze_umaze_goal_2
    pointmaze/pointmaze_umaze_goal_3
)
ACTOR_LRS=(1e-7 1e-4)

for seed in 0 1 2; do
    for env in "${ENVS[@]}"; do
        for actor_lr in "${ACTOR_LRS[@]}"; do
            sbatch --export=ALL,SEED="$seed",ENV="$env",MODEL_PATH="$MODEL_PATH",ACTOR_LR="$actor_lr" "$INNER"
        done
    done
done
