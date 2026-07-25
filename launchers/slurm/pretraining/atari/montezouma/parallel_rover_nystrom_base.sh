#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua-longrun

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate dist_matching

export HYDRA_FULL_ERROR=1




PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pretrain_parallel.py \
    --config-name=pretrain_parallel/pretrain_montezouma \
    wandb_tag=parallel \
    use_wandb=true \
    seed=${SEED} \
    agent.feature_dim=${FEATURE_DIM} \
    agent.batch_size_actor=${BATCH_SIZE_ACTOR} \
    agent.subsamples=${SUBSAMPLE} \
    agent.kernel_bandwidth_mult=${KERNEL_BANDWIDTH_MULTIPLIER} \
    agent.sink_schedule=0.0 \
    agent.encoded_fifo_capacity=${BATCH_SIZE_ACTOR} \
    wandb_project=montezouma_hp

