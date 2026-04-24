#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate dist_matching


export HYDRA_FULL_ERROR=1

# if subsamples is none, we set the SUBSAMPLES equal to null
if [ "$SUBSAMPLES" = "none" ]; then
    SUBSAMPLES=null
    EXTRA_ARGS="agent.lambda_reg=1e-6"
fi

#   0 -> linear(0.0, 0.05, 50_000)
#   1 -> linear(0.0, 0.001, 50_000)
#   2 -> linear(0.0, 1, 50_000)
#   3 -> linear(0.0, 0.0001, 500_000)
#   4 -> linear(1.0, 1.0,    100_000)
sink_schedules=(
    "linear(0.0, 0.05, 50_000)"
    "linear(0.0, 0.001, 50_000)"
    "linear(0.0, 1, 50_000)"
    "linear(0.0, 0.0001, 500_000)"
    "linear(1.0, 1.0,    100_000)"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"
    
python pretrain.py --config-name=pretrain/pretrain_rover_multiplerooms agent.embeddings=true discount=0.99 agent=rover_nystrom agent.subsamples=${SUBSAMPLES} agent.lr_actor=${LR_ACTOR} agent.pmd_steps=100 agent.feature_dim=${FEATURE_DIM} num_seed_frames=4000 obs_type=pixels agent.update_every_steps=50 agent.batch_size=512 agent.batch_size_actor=10_000 device=cuda seed=${SEED} use_wandb=true wandb_project="rover_nystrom" "agent.sink_schedule='${SINK_SCHEDULE}'" ${EXTRA_ARGS} 