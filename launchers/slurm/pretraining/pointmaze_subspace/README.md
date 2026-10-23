# PointMaze subspace experiments

Every experiment directory contains only `launcher.sh` and `base.sh`:

```text
pointmaze_subspace/
  umaze/
    states/{rover,baselines}/
    pixels/{rover,baselines}/
  largedense/
    states/{rover,baselines}/
    pixels/{rover,baselines}/
```

Run launchers from the repository root. Each launcher submits its own base
script directly with `sbatch`. Edit `seeds=(1)` and sweep arrays in the launcher.
There are no shared shell helpers or Python submission scripts.

ROVER launchers use nested loops over values listed in each launcher.
States and pixels both sweep dynamic kernel type (`inner_product` and
`gaussian`); coverage kernel remains Gaussian. Current arrays submit 8 UMaze
states, 32 LargeDense states, 4 UMaze pixels, and 32 LargeDense pixels jobs
per seed.

| Parameter | States reference | States alternative | Pixels reference | Pixels alternative |
|---|---:|---:|---:|---:|
| `batch_size_actor` | 16000 | 32000 | 25000 | 16000 |
| `feature_dim` | 256 | 512 | 2048 | 1024 |
| `subsamples` | 4000 | 8000 | 10000 | 4000 |
| Sink index | 0 | 1 | 0 | 1 |

States sink 0 is `linear(0.0,0.8,250000)`; sink 1 is
`linear(0.0,1.0,300000,750000)`. Pixels use the opposite ordering to preserve
their reference schedule. `pca_truncation` follows `subsamples`.

The states profile uses `learned_xy_fast` hyperparameters: state observations,
trained state features for inner-product dynamics, fixed raw XY features for
Gaussian state coverage, encoder updates every 10 steps, and `lambda_reg=1e-6`.
Coverage features are independent of the trained dynamics encoder.

The pixels profile uses `good_pixels` hyperparameters: four grayscale frames,
trained CNN dynamics with an inner-product kernel and a separate frozen CNN
for Gaussian pixel coverage. Coverage uses the newest frame. It retains
`lambda_reg=1e-3`, actor learning rate 25, encoder updates every 50 steps,
and 1000 warm-up updates. Cholesky tolerance stays at the reference values:
0 for states, 1e-6 for pixels; it is not swept. Image CNN ignores `hidden_dim`.

UMaze and LargeDense use their full `(x,y,vx,vy)` observation environment
configs. UMaze launchers sweep start positions `[-1,1]`, `[1,0]`, and `[1,1]`;
LargeDense keeps fixed start `[-4,3]`. Baselines match the ROVER
profile for each input: states use continuing tasks; pixels retain the
episodic setting of `good_pixels`. Both use a 10000-frame seed phase.

All runs train for 1000000 aggregate environment frames. Coverage evaluation
and trajectory images run every 20000 frames, including a final evaluation
at 1000000. Coverage uses 50 trajectories, a 90×90 grid, radius 0.08, and the
existing exploratory evaluation policy. Initial trajectory positions are red.

Numbered checkpoints are saved at 250000, 500000, 750000, and 1000000 frames.
`final_snapshot.pt` is saved after final evaluation. Best coverage selection
ignores evaluations before `num_seed_frames` and writes `best_snapshot.pt`
plus `best_snapshot_step.txt` containing step, frame, and coverage. The
`save_snapshot=false` setting disables repeated latest-checkpoint writes;
explicit numbered, best, and final checkpoints remain enabled.

## Cluster submission

For UMaze:

```bash
bash launchers/slurm/pretraining/pointmaze_subspace/umaze/states/rover/launcher.sh
bash launchers/slurm/pretraining/pointmaze_subspace/umaze/pixels/rover/launcher.sh
bash launchers/slurm/pretraining/pointmaze_subspace/umaze/states/baselines/launcher.sh
bash launchers/slurm/pretraining/pointmaze_subspace/umaze/pixels/baselines/launcher.sh
```

For LargeDense:

```bash
bash launchers/slurm/pretraining/pointmaze_subspace/largedense/states/rover/launcher.sh
bash launchers/slurm/pretraining/pointmaze_subspace/largedense/pixels/rover/launcher.sh
bash launchers/slurm/pretraining/pointmaze_subspace/largedense/states/baselines/launcher.sh
bash launchers/slurm/pretraining/pointmaze_subspace/largedense/pixels/baselines/launcher.sh
```

Each baseline launcher submits `rnd_discrete`, `icm_apt_discrete`,
`maxent_discrete`, `cic_discrete`, and `smm_discrete` once per seed.

## Local LargeDense checks

Activate `dist_matching` first. These checks use the exact reference launcher
parameters: actor batches 16000/25000, feature dimensions 256/2048, landmarks
4000/10000, seed phase 10000, and 50 evaluation trajectories. Both dynamics
encoders train; both coverage representations stay fixed. Only training
length, checkpoint times, output directory, and W&B logging differ.
The encoder learning-rate schedule retains its 1M-frame horizon.
Use `device=cpu` if CUDA is unavailable.

States in dynamics and coverage:

```bash
python pretrain_parallel.py \
  --config-name=pretrain_parallel/pointmaze_subspace_states \
  env=pointmaze/pointmaze_largedense_full_state_discrete \
  num_train_frames=40000 agent.total_train_steps=1000000 \
  use_wandb=false device=cuda seed=1 \
  'snapshots=[20000,40000]' \
  'hydra.run.dir=./exp_local_parallel/pointmaze_subspace_smoke/largedense_states_${now:%Y%m%d_%H%M%S_%f}'
```

Pixels in dynamics and coverage:

```bash
python pretrain_parallel.py \
  --config-name=pretrain_parallel/pointmaze_subspace_pixels \
  env=pointmaze/pointmaze_largedense_full_state_discrete \
  num_train_frames=40000 agent.total_train_steps=1000000 \
  use_wandb=false device=cuda seed=1 \
  'snapshots=[20000,40000]' \
  'hydra.run.dir=./exp_local_parallel/pointmaze_subspace_smoke/largedense_pixels_${now:%Y%m%d_%H%M%S_%f}'
```

The checks should produce evaluation images at 20000 and 40000 frames,
numbered/final checkpoints, and an eligible best snapshot with its timestep
text file. For full local training, set `num_train_frames=1000000` and
`snapshots=[250000,500000,750000,1000000]`.
