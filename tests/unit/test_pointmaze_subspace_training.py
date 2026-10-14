"""Regression checks for evaluation and checkpoint timing, without training."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pretrain
import pretrain_parallel
from agent.utils_debug_visualization import CoverageProgress


@pytest.fixture(params=[pretrain, pretrain_parallel], ids=['sequential', 'parallel'])
def workspace(request, tmp_path):
    module = request.param
    ws = module.Workspace.__new__(module.Workspace)
    ws.cfg = SimpleNamespace(
        seed=1, domain='gym', task_name='PointMaze_UMaze-v3',
        obs_type='states', action_repeat=1, num_train_frames=4,
        num_seed_frames=10000, eval_every_frames=2, num_eval_episodes=1,
        coverage_eval_enabled=True, coverage_num_trajectories=1,
        plot_eval_trajectories=False, snapshot_dir='models',
    )
    ws.work_dir = tmp_path
    ws._global_step = 0
    ws._global_episode = 0
    ws._coverage_progress = CoverageProgress()
    ws._best_snapshot_coverage = None
    ws.save_eval_best = True
    ws.save_snapshot_flag = False
    ws.snapshot_steps = [2, 4]
    ws.agent = SimpleNamespace(
        training=False, train=lambda mode: None,
        init_meta=lambda: {}, update_meta=lambda meta, *args, **kwargs: meta,
        act=lambda *args, **kwargs: 0,
        act_parallel=lambda *args, **kwargs: np.zeros(2, dtype=np.int64),
    )
    ws.logger = MagicMock()
    ws.timer = MagicMock()
    ws.video_recorder = MagicMock()
    ws.train_video_recorder = MagicMock()
    ws.replay_storage = MagicMock()
    ws._log_montezuma_episode_metrics = MagicMock()
    ws._evaluation_rng_state = MagicMock(return_value={})
    ws._restore_evaluation_rng_state = MagicMock()
    ws._resume_signature = lambda: {}
    return module, ws


def test_best_selection_ignores_pre_seed_peak(workspace):
    module, ws = workspace
    ws.save_snapshot = MagicMock()
    ws.eval_env = MagicMock()
    ws.eval_env.reset.return_value = SimpleNamespace(last=lambda: True)
    with (
        patch.object(module, 'extract_eval_trajectory_point', return_value=np.zeros(2)),
        patch.object(module, 'pointmaze_free_space_coverage') as coverage,
        patch.object(module.utils, 'set_seed_everywhere'),
    ):
        for frame, pct in [(0, 90.0), (10000, 20.0), (20000, 15.0), (40000, 30.0)]:
            ws._global_step = frame
            coverage.return_value = (1, 1, pct)
            ws.eval()
            assert ws.save_snapshot.call_count == {0: 0, 10000: 1, 20000: 1, 40000: 2}[frame]
    assert ws._best_snapshot_coverage == 30.0
    assert ws._coverage_progress.best_coverage == 90.0


def test_best_snapshot_records_current_timestep(workspace):
    module, ws = workspace
    ws._global_step = 20000
    ws._best_snapshot_coverage = 40.0
    ws._coverage_progress.update(40.0, ws.global_frame)
    with patch.object(module.torch, 'save') as save:
        ws.save_snapshot(filename='best_snapshot.pt', force=True)
    text = (ws.work_dir / 'models/best_snapshot_step.txt').read_text()
    assert 'global_step=20000\n' in text
    assert 'global_frame=20000\n' in text
    assert 'coverage_pct=40.0\n' in text
    assert save.call_args.args[0]['_best_snapshot_coverage'] == 40.0


def test_final_evaluation_and_checkpoints_do_not_need_episode_end(workspace):
    module, ws = workspace
    time_step = SimpleNamespace(
        observation=np.zeros(4), image_observation=None,
        reward=0.0, last=lambda: False,
    )
    ws.train_env = MagicMock()
    if module is pretrain_parallel:
        ws.num_envs = 2
        ws.base_seed = 1
        ws.agent_requires_replay = False
        ws._assert_update_schedule_covered = MagicMock()
        ws._drain_encoded_actor_fifo_if_due = MagicMock()
        observations = np.zeros((2, 4))
        infos = {'time_step': [time_step, time_step]}
        ws.train_env.reset.return_value = (observations, infos)
        ws.train_env.step.return_value = (
            observations, np.zeros(2), np.zeros(2, dtype=bool),
            np.zeros(2, dtype=bool), infos,
        )
    else:
        ws.train_env.reset.return_value = time_step
        ws.train_env.step.return_value = time_step

    evaluated = []
    saved = []
    ws.eval = lambda: evaluated.append(ws.global_frame)

    def save_snapshot(filename=None, force=False):
        saved.append((ws.global_frame, filename))
        if filename is None:
            ws.snapshot_steps.pop(0)

    ws.save_snapshot = save_snapshot
    ws.train()
    assert evaluated == [0, 2, 4]
    assert saved == [(2, None), (4, None), (4, 'final_snapshot.pt')]
    assert ws.snapshot_steps == []
