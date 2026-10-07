#!/usr/bin/env bash
# Paired image/state runs after the three correctness fixes; sequential on one GPU.
set -euo pipefail

ROVER_VERIFY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$ROVER_VERIFY_ROOT"
ROVER_VERIFY_PYTHON=${ROVER_VERIFY_PYTHON:-/home/mprattico-iit.local/miniconda3/envs/dist_matching/bin/python}
ROVER_VERIFY_FRAMES=${ROVER_VERIFY_FRAMES:-200000}
export MUJOCO_GL=egl
if (( $# == 0 )); then
    set -- 1 2 3
fi

shared=(
    "num_train_frames=$ROVER_VERIFY_FRAMES"
    num_envs=4 num_seed_frames=10000 eval_every_frames=20000
    '+coverage_eval_enabled=true' '+coverage_num_trajectories=50'
    '+coverage_grid_size=90' '+coverage_radius=0.08'
    '+coverage_expansion_tolerance=0.25'
    'snapshots=[50000,100000,200000]'
    save_snapshot=true save_eval_best=false use_wandb=false
    parallel_attach_eval_env_for_debug=true
    agent.embeddings=true agent.feature_learning_loss=infonce
    agent.linear_projection=true agent.coverage_embeddings=false
    agent.kernel_type=gaussian agent.kernel_bandwidth=null
    agent.kernel_bandwidth_mult=1.0 agent.coverage_kernel_bandwidth=null
    agent.coverage_kernel_bandwidth_mult=1.0
    agent.lambda_reg=1e-6 agent.lr_actor=10
    'agent.sink_schedule="linear(0.0,0.8,250000)"'
)

for rover_verify_seed in "$@"; do
    for rover_verify_mode in image state; do
        if [[ "$rover_verify_mode" == image ]]; then
            rover_verify_config=pretrain_parallel/pretrain_pointmaze_umaze_1_pixels_subspace
        else
            rover_verify_config=pretrain_parallel/pretrain_pointmaze_umaze_1_subspace
        fi
        rover_verify_dir="exp_local_parallel/$(date +%Y.%m.%d/%H%M%S_%N)_fixed_${rover_verify_mode}_s${rover_verify_seed}"
        command=("$ROVER_VERIFY_PYTHON" pretrain_parallel.py
            "--config-name=$rover_verify_config" "${shared[@]}"
            "seed=$rover_verify_seed" "hydra.run.dir=$rover_verify_dir")
        if [[ ${ROVER_VERIFY_DRY_RUN:-0} == 1 ]]; then
            # Resolve with Hydra without creating environments or starting training.
            "${command[@]}" --cfg job > /dev/null
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            "${command[@]}"
        fi
    done
done
