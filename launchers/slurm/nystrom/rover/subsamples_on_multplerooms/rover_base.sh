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
    EXTRA_ARGS="agent.lambda_reg=1e-3"
fi
    
python pretrain.py --config-name=pretrain/pretrain_rover_multiplerooms agent.embeddings=true discount=0.99 agent=rover_nystrom agent.subsamples=${SUBSAMPLES} agent.pmd_steps=100 agent.feature_dim=109 num_seed_frames=4000 obs_type=pixels agent.update_every_steps=50 agent.batch_size=512 agent.batch_size_actor=10_000 device=cuda seed=${SEED} use_wandb=true wandb_project="rover_nystrom" ${EXTRA_ARGS} 