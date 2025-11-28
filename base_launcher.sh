#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=cpu

cd $SLURM_SUBMIT_DIR


export HYDRA_FULL_ERROR=1


# Load environment
source ~/.bashrc
conda activate dist_matching

python pretrain.py agent=dist_matching env.render_mode=null save_video=false num_train_frames=104000 configs/env=four_rooms10_2