#!/bin/bash

seeds="0 1 2 3 4 5 6"

envs=(
    "gridworld/maze_108_seed7_env_3"
)

actor_lrs=(
    "1e-4"
    "1e-7"
)

obs_types=(
    "pixels"
)

agent_specs=(
    "ddpg_discrete|null"
    "cic_discrete|models/maze/108/pixels/cic/snapshot.pt"
    "icm_apt_discrete|models/maze/108/pixels/icm_apt/snapshot.pt"
    "rnd_discrete|models/maze/108/pixels/rnd/snapshot.pt"
    "smm_discrete|models/maze/108/pixels/smm/snapshot.pt"
)

for seed in $seeds; do
    for env in "${envs[@]}"; do
        for spec in "${agent_specs[@]}"; do
            IFS='|' read -r agent path <<< "${spec}"
            for lr in "${actor_lrs[@]}"; do
                for obs in "${obs_types[@]}"; do
                    sbatch --export=SEED="${seed}",ENV="${env}",AGENT="${agent}",MODEL_PATH="${path}",ACTOR_LR="${lr}",OBS_TYPE="${obs}" launchers/slurm/finetuning/maze/108/baselines/baseline_base.sh
                done
            done
        done
    done
done
