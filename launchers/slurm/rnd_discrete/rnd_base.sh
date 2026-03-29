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

# If ENV starts with "montezouma", enable non-episodic intrinsic returns
if [[ "${ENV}" == montezouma* ]]; then
	EXTRA_FLAGS=' --config-name=pretrain/pretrain_montezouma agent.non_episodic_intrinsic_returns=true'
else
	EXTRA_FLAGS=''
fi

python pretrain.py $EXTRA_FLAGS agent=rnd_discrete use_wandb=true eval_every_frames=20_000 num_train_frames=1_000_000 env=${ENV} device=cuda seed=${SEED} wandb_tag="rnd_discrete" obs_type=${OBS_TYPE} env.render_mode="rgb_array" wandb_project="url_atari_baselines" agent.feature_dim=${FEATURE_DIM}
