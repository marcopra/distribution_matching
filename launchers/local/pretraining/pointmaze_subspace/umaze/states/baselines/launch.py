#!/usr/bin/env python3
"""Launch a 45-run PointMaze baseline sweep in eight detached GPU queues."""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[7]
PYTHON = Path.home() / 'miniconda3/envs/dist_matching/bin/python'
AGENTS = ['rnd_discrete', 'icm_apt_discrete', 'maxent_discrete',
          'cic_discrete', 'smm_discrete']


def command(run, batch, obs_type, num_train_frames):
    directory = batch / 'runs' / run['name']
    frame_stack = 4 if obs_type == 'pixels' else 1
    if num_train_frames == 250_000:
        snapshots = '[100000,250000]'
    elif num_train_frames == 1_000_000:
        snapshots = '[250000,500000,750000,1000000]'
    else:
        snapshots = f'[{num_train_frames}]'
    return [str(PYTHON), '-u', str(ROOT / 'pretrain.py'),
            '--config-name=pretrain/pointmaze_subspace_baselines',
            'env=pointmaze/pointmaze_umaze_full_state_discrete',
            f"env.pointmaze.start_position={json.dumps(run['start'], separators=(',', ':'))}",
            'env.continuing_task=false' if obs_type == 'pixels' else 'env.continuing_task=true',
            f'obs_type={obs_type}', f'frame_stack={frame_stack}',
            f"agent={run['agent']}", f"seed={run['seed']}", 'device=cuda',
            f'num_train_frames={num_train_frames}', 'eval_every_frames=20000',
            'plot_eval_trajectories=true', 'save_eval_best=true',
            'save_snapshot=false', f'snapshots={snapshots}',
            f'snapshot_dir={directory / "models"}', 'use_wandb=true',
            'wandb_project=pointmaze_subspace',
            f"wandb_tag={run['agent']}_umaze_{obs_type}_local{num_train_frames // 1000}k",
            f"wandb_run_name={run['name']}", f'hydra.run.dir={directory}']


def worker(batch, gpu, slot=None):
    manifest = json.loads((batch / 'manifest.json').read_text())
    obs_type = manifest['obs_type']
    num_train_frames = manifest['num_train_frames']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='2', HYDRA_FULL_ERROR='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               WANDB_RUN_GROUP=manifest.get('wandb_group', batch.name),
               WANDB_JOB_TYPE='local-pretraining',
               WANDB_MODE='online', PYTHONUNBUFFERED='1')
    cpus = manifest['cpu_affinity'][str(gpu)]
    os.sched_setaffinity(0, cpus)
    for run in manifest['runs']:
        if run['gpu'] != gpu or (slot is not None and run.get('slot') != slot):
            continue
        status = batch / 'status' / (run['name'] + '.json')
        if status.exists() and json.loads(status.read_text())['state'] == 'completed':
            continue
        record = dict(run, state='running', started=datetime.now().isoformat())
        status.write_text(json.dumps(record, indent=2))
        print(f"START GPU {gpu}: {run['name']}", flush=True)
        with (batch / 'logs' / (run['name'] + '.log')).open('a') as log:
            result = subprocess.run(command(run, batch, run.get('obs_type', obs_type), num_train_frames), cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        record.update(state='completed' if result.returncode == 0 else 'failed',
                      returncode=result.returncode, finished=datetime.now().isoformat())
        status.write_text(json.dumps(record, indent=2))
        print(f"{record['state'].upper()} GPU {gpu}: {run['name']} ({result.returncode})",
              flush=True)
    print(f'GPU {gpu} queue finished.', flush=True)


def launch(obs_type, num_train_frames, starts, run_specs=None, batch_name=None,
           wandb_group=None):
    import torch
    assert torch.cuda.device_count() == 8, 'This allocation expects eight GPUs.'
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 32
    if run_specs is None:
        run_specs = [dict(obs_type=obs_type, seed=seed, start=start, agent=agent)
                     for seed in [11, 12, 13] for start in starts for agent in AGENTS]
    run_obs_types = {spec.get('obs_type', obs_type) for spec in run_specs}
    launch_label = next(iter(run_obs_types)) if len(run_obs_types) == 1 else 'mixed'
    name = batch_name or (f'umaze_baselines_{launch_label}_{num_train_frames // 1000}k_' +
                          datetime.now().strftime('%Y%m%d_%H%M%S'))
    batch = ROOT / 'exp_local' / name
    batch_group = wandb_group or batch.name
    for name in ['logs', 'status', 'runs']:
        (batch / name).mkdir(parents=True, exist_ok=True)
    runs = []
    for spec in run_specs:
        current_obs = spec.get('obs_type', obs_type)
        start = list(spec['start'])
        name = (f'umaze_{current_obs}_{spec["agent"]}_start{start[0]:g}_{start[1]:g}'
                f'_seed{spec["seed"]}_{num_train_frames // 1000}k')
        runs.append(dict(name=name, obs_type=current_obs, agent=spec['agent'],
                         seed=spec['seed'], start=start, gpu=len(runs) % 8))
    assert runs and len(runs) == len({r['name'] for r in runs})
    manifest = dict(runs=runs, obs_type=launch_label, num_train_frames=num_train_frames,
                    wandb_group=batch_group,
                    cpu_affinity={str(g): cpus[4*g:4*g+4] for g in range(8)},
                    gpu_names=[torch.cuda.get_device_name(g) for g in range(8)])
    (batch / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (batch / 'commands.json').write_text(json.dumps([command(r, batch, r['obs_type'], num_train_frames) for r in runs], indent=2))
    for gpu in range(8):
        session = f'umaze{launch_label}{num_train_frames // 1000}k_{batch.name.rsplit("_", 1)[-1]}_gpu{gpu}'
        cmd = shlex.join([str(PYTHON), '-u', str(Path(__file__).resolve()),
                          '--worker', str(gpu), '--batch', str(batch)])
        cmd += ' >> ' + shlex.quote(str(batch / 'logs' / f'queue_gpu{gpu}.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, cmd], check=True)
    print(batch)
    print(f'{launch_label} observations, {num_train_frames} training frames')
    print('Runs per GPU:', dict(Counter(r['gpu'] for r in runs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--slot', type=int,
                        help='Optional queue slot used to run multiple state jobs on one GPU')
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--obs-type', choices=['states', 'pixels'], default='states')
    parser.add_argument('--num-train-frames', type=int, default=500_000)
    parser.add_argument('--starts-json', default='[[-1,1],[1,1],[1,0]]',
                        help='JSON array of PointMaze start coordinates')
    parser.add_argument('--run-specs-json',
                        help='Optional JSON array of {obs_type, agent, seed, start} runs')
    parser.add_argument('--run-specs-file', type=Path,
                        help='Optional JSON file of {obs_type, agent, seed, start} runs')
    parser.add_argument('--batch-name',
                        help='Optional shared batch directory and W&B group name')
    args = parser.parse_args()
    if args.worker is None:
        starts = json.loads(args.starts_json)
        if not starts or any(len(start) != 2 for start in starts):
            parser.error('--starts-json must contain pairs of coordinates')
        if args.run_specs_json and args.run_specs_file:
            parser.error('choose only one of --run-specs-json and --run-specs-file')
        run_specs = json.loads(args.run_specs_json) if args.run_specs_json else None
        if args.run_specs_file:
            run_specs = json.loads(args.run_specs_file.read_text())
        launch(args.obs_type, args.num_train_frames, starts, run_specs,
               batch_name=args.batch_name)
    else:
        worker(args.batch.resolve(), args.worker, args.slot)
