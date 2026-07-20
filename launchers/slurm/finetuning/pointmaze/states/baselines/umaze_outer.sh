#!/bin/bash
set -euo pipefail

INNER="launchers/slurm/finetuning/pointmaze/states/baselines/umaze_inner.sh"
ENVS=(
    pointmaze/pointmaze_umaze_goal_1
    pointmaze/pointmaze_umaze_goal_2
    pointmaze/pointmaze_umaze_goal_3
)
SPECS=(
    "ddpg_discrete|none|10000"
    "cic_discrete|/home/mprattico-iit.local/distribution_matching/models/pointmaze/umaze/states/cic/models/states/gym/cic/0/snapshot_1000200.pt|30000"
    "icm_apt_discrete|/home/mprattico-iit.local/distribution_matching/models/pointmaze/umaze/states/icm_apt/models/states/gym/icm_apt/0/snapshot_1000200.pt|30000"
    "rnd_discrete|/home/mprattico-iit.local/distribution_matching/models/pointmaze/umaze/states/rnd/models/states/gym/rnd/0/snapshot_1000200.pt|30000"
    "smm_discrete|/home/mprattico-iit.local/distribution_matching/models/pointmaze/umaze/states/smm/models/states/gym/smm/0/snapshot_1000200.pt|30000"
    "maxent_discrete|/home/mprattico-iit.local/distribution_matching/models/pointmaze/umaze/states/maxent/models/states/gym/maxent/0/snapshot_500100.pt|30000"
)

for seed in 0 1 2; do
    for env in "${ENVS[@]}"; do
        for spec in "${SPECS[@]}"; do
            IFS='|' read -r agent model_path actor_update_threshold <<< "$spec"
            sbatch --export=ALL,SEED="$seed",ENV="$env",AGENT="$agent",MODEL_PATH="$model_path",ACTOR_UPDATE_THRESHOLD="$actor_update_threshold" "$INNER"
        done
    done
done
