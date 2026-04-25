#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

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

# Decode sink_schedule from index to avoid quoting issues in sbatch --export
sink_schedules=(
    "linear(0.0, 0.0001, 1_000_000)"
    "linear(0.0, 0.001,  2_000_000)"
    "linear(0.0, 1,      1_000_000)"
    "linear(0.0, 0.0001, 500_000)"
    "linear(1.0, 1.0,    100_000)"
)
SINK_SCHEDULE="${sink_schedules[$SINK_IDX]}"

    

python pretrain.py \
    --config-name=pretrain/pretrain_pointmaze_umaze_1 \
    agent=rover_nystrom_memory_efficient \
    use_wandb=true \
    agent.subsamples=${SUBSAMPLES} \
    agent.lr_actor=${LR_ACTOR} \
    agent.feature_dim=${FEATURE_DIM}\
    seed=${SEED} \
    "agent.sink_schedule='${SINK_SCHEDULE}'" \
    env=${ENV} \
    agent.update_every_steps=100 \
    agent.pmd_steps=100 \
    ${EXTRA_ARGS} 

