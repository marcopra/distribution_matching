import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import inspect

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import wandb
from dm_env import specs
import gym_env

import utils
from logger import Logger
from replay_buffer_parallel import ReplayBufferStorageParallel, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import ale_py
from omegaconf import open_dict
from agent.rover_visualization.domains import save_maze_trajectory_overlay_plot


torch.backends.cudnn.benchmark = True


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        for stream in self.streams:
            if hasattr(stream, "fileno"):
                return stream.fileno()
        raise OSError("no stream has fileno")

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", None)


class ConsoleLog:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.log_file = open(self.log_path, 'a', buffering=1)
        sys.stdout = Tee(self.stdout, self.log_file)
        sys.stderr = Tee(self.stderr, self.log_file)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()


def enable_console_log(log_path):
    return ConsoleLog(log_path)


class NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False



def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape if obs_spec.shape else (1,)

    # Determine mode based on action spec
    if hasattr(action_spec, 'num_values'):
        # Discrete action space
        cfg.action_shape = (action_spec.num_values,)
    else:
        # Continuous action space
        cfg.action_shape = action_spec.shape

    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        if not hasattr(self.cfg, 'grayscale'):
            with open_dict(self.cfg):
                self.cfg.grayscale = False
        if not hasattr(self.cfg, 'num_envs'):
            with open_dict(self.cfg):
                self.cfg.num_envs = 1
        if not hasattr(self.cfg, 'base_seed'):
            with open_dict(self.cfg):
                self.cfg.base_seed = self.cfg.seed
        if cfg.seed == -1:
            cfg.seed = np.random.randint(0, 1000000)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            if cfg.wandb_id is not None and cfg.wandb_id != "none":
                wandb.init(
                    id=cfg.wandb_id,
                    resume='must',
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=False,
                    mode=getattr(cfg, 'wandb_mode', 'online'))
            else:
                wandb.init(
                    config=OmegaConf.to_container(cfg, resolve=True),
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=False,
                    mode=getattr(cfg, 'wandb_mode', 'online'))

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        if cfg.use_wandb:
            wandb.define_metric('train/frame')
            wandb.define_metric('train/*', step_metric='train/frame')
        # create envs
        env_kwargs = OmegaConf.to_container(cfg.env, resolve=True) if hasattr(cfg, 'env') else {}
        env_kwargs.pop('name', None)
        env_kwargs.pop('synthetic_first_transition', None)

        self.num_envs = int(getattr(self.cfg, "num_envs", 1))
        self.base_seed = int(getattr(self.cfg, "base_seed", self.cfg.seed))*self.num_envs
        self.parallel_attach_eval_env_for_debug = bool(
            getattr(self.cfg, "parallel_attach_eval_env_for_debug", False)
        )

        self.collection_env = gym_env.make_async_vector_env(
            self.num_envs,
            self.base_seed,
            self.cfg.task_name,
            self.cfg.obs_type,
            frame_stack=self.cfg.frame_stack,
            action_repeat=self.cfg.action_repeat,
            resolution=self.cfg.resolution,
            grayscale=self.cfg.grayscale,
            url=True,
            **env_kwargs,
        )
        self.train_env = gym_env.make_async_vector_env(
            self.num_envs,
            self.base_seed,
            self.cfg.task_name,
            self.cfg.obs_type,
            frame_stack=self.cfg.frame_stack,
            action_repeat=self.cfg.action_repeat,
            resolution=self.cfg.resolution,
            grayscale=self.cfg.grayscale,
            url=False,
            **env_kwargs,
        )
        self.eval_env = gym_env.make(
            self.cfg.task_name,
            self.cfg.obs_type,
            frame_stack=self.cfg.frame_stack,
            action_repeat=self.cfg.action_repeat,
            seed=self.cfg.seed,
            resolution=self.cfg.resolution,
            grayscale=self.cfg.grayscale,
            url=False,
            **env_kwargs,
        )

        # TODO: modify the make function to work with cfg and modify inplace the cfg values, this is a temporary solution to avoid modifying the make function
        if isinstance(self.eval_env.unwrapped, ale_py.env.AtariEnv) or str(self.cfg.task_name).startswith("ALE/"):
            # L'action repeat è gestito internamente da ALE, quindi forziamo action_repeat a 1
            with open_dict(self.cfg):
                self.cfg.action_repeat = 1
        # Get observation and action specs for the agent
        obs_spec = gym_env.observation_spec(self.eval_env)
        action_spec = gym_env.action_spec(self.eval_env)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                obs_spec,
                                action_spec,
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)
        self.agent_requires_replay = bool(
            getattr(self.agent, 'requires_replay', True)
        )

        if int(getattr(cfg, 'snapshot_ts', 0)) > 0:
            payload = self.load_pretrained_snapshot()
            self.agent.init_from(payload['agent'])
        elif getattr(cfg, 'p_path', None) not in (None, 'none'):
            if str(cfg.p_path).endswith('.npy'):
                self.agent = utils.load_policy_weights_into_agent(
                    self.agent, cfg.p_path, device=self.device
                )
            else:
                payload = self.load_snapshot_from_path(cfg.p_path)
                self.agent.init_from(payload['agent'])

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        time_step = self.eval_env.reset()

        if hasattr(self.agent, 'insert_env'):
            self.agent.insert_env(self.eval_env)

        # create replay buffer
        data_specs = (obs_spec,
                      action_spec,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      )

        # create data storage
        self.replay_storage = ReplayBufferStorageParallel(
            data_specs,
            meta_specs,
            self.work_dir / 'buffer',
            num_envs=self.num_envs,
            retain_episodes=self.agent_requires_replay or cfg.save_buffer,
        )

        # create replay buffer
        first_transition = type(self.agent).__name__ == 'RoverAgent'
        transition_view = bool(
            getattr(self.agent, 'requires_transition_view', False)
        )
        self.replay_loader = None
        if self.agent_requires_replay:
            self.replay_loader = make_replay_loader(
                self.replay_storage,
                cfg.replay_buffer_size,
                cfg.batch_size,
                cfg.replay_buffer_num_workers,
                cfg.save_buffer,
                cfg.nstep,
                cfg.discount,
                first_transition=first_transition,
                transition_view=transition_view,
                max_pending_transitions=getattr(self.agent, 'max_pending_transitions', None),
                drop_oldest_pending_on_overflow=(
                    getattr(self.agent, 'max_pending_transitions', None) is not None
                ),
            )

        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb,
            grayscale=self.cfg.grayscale,
            is_training_sample=False)

        self.snapshot_steps = cfg.snapshots
        self.save_snapshot_flag =  cfg.save_snapshot if hasattr(cfg, 'save_snapshot') else True

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._last_train_log_frame = 0
        self._seed_trajectories = [[] for _ in range(self.num_envs)]
        self._completed_seed_trajectories = []
        self._seed_trajectories_logged = False

    @property
    def is_montezuma(self):
        return 'MontezumaRevenge' in str(self.cfg.task_name)

    def _get_time_step_info(self, time_step):
        info = getattr(time_step, 'info', None)
        return info if isinstance(info, dict) else {}

    def _log_montezuma_episode_metrics(self, log, time_step, ty='train'):
        if not self.is_montezuma:
            return
        info = self._get_time_step_info(time_step)
        for info_key, metric_key in (
            ('montezuma_escaped_first_room', 'escaped_first_room'),
            ('montezuma_unique_rooms_visited', 'unique_rooms_visited'),
            ('montezuma_room_transition_count', 'room_transition_count'),
        ):
            if info_key in info:
                log(metric_key, float(info[info_key]))
        self.logger.log_room_route(
            info.get('montezuma_room_route'),
            self.global_episode,
            self.global_frame,
            ty,
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self.replay_loader is None:
            return None
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _should_use_synthetic_first_transition(self):
        env_cfg = getattr(self.cfg, "env", None)
        if env_cfg is not None and hasattr(env_cfg, "synthetic_first_transition"):
            return bool(env_cfg.synthetic_first_transition)
        return str(self.cfg.task_name) in {"MiddleRoom-v0"}

    def _maybe_set_synthetic_first_transition(self, time_step, meta):
        if not self._should_use_synthetic_first_transition():
            return
        self.replay_storage.set_synthetic_first_transition(time_step, meta=meta)

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        eval_mode = True
        while eval_until_episode(episode):
            meta = self.agent.init_meta()
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=eval_mode)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if self.is_montezuma and episode > 0:
                self._log_montezuma_episode_metrics(log, time_step, ty='eval')

    def _time_steps_from_infos(self, infos, required_mask=None):
        time_steps = infos.get("time_step")
        if time_steps is None:
            raise RuntimeError("Vector env did not return ExtendedTimeStep objects in infos['time_step']")
        if required_mask is None:
            required_mask = np.ones(len(time_steps), dtype=bool)
        return [
            time_steps[env_id] if required_mask[env_id] else None
            for env_id in range(len(required_mask))
        ]

    def _reset_done_envs(self, env, done_mask):
        try:
            return env.reset(options={"reset_mask": done_mask.astype(np.bool_)})
        except (TypeError, AssertionError, NotImplementedError):
            if np.all(done_mask):
                return env.reset()
            raise RuntimeError(
                "This Gymnasium AsyncVectorEnv does not support partial reset_mask; "
                "upgrade Gymnasium or use num_envs=1."
            )

    def _update_agent_once(self, logical_step):
        update_signature = inspect.signature(self.agent.update)
        if 'replay_buffer' in update_signature.parameters:
            metrics = self.agent.update(
                self.replay_iter,
                int(logical_step),
                replay_buffer=self.replay_loader.dataset,
            )
        else:
            metrics = self.agent.update(self.replay_iter, int(logical_step))
        self.logger.log_metrics(metrics, self.global_frame, ty='train')
        return metrics

    def _pending_transition_count(self):
        if self.replay_loader is None:
            return 0
        replay_dataset = self.replay_loader.dataset
        if not hasattr(replay_dataset, "pending_transition_count"):
            return 0
        return int(replay_dataset.pending_transition_count())

    def _drain_encoded_actor_fifo_if_due(self, logical_steps, force=False):
        if not self.agent_requires_replay or self.replay_loader is None:
            return False
        if not hasattr(self.agent, "drain_encoded_actor_fifo"):
            return False

        if not force:
            update_every_steps = int(getattr(self.agent, "update_every_steps", 1))
            force = any(int(step) % update_every_steps == 0 for step in logical_steps)
        if not force:
            max_pending = getattr(self.agent, "max_pending_transitions", None)
            if max_pending is not None:
                force = self._pending_transition_count() >= int(max_pending)
        if not force:
            return False

        return bool(self.agent.drain_encoded_actor_fifo(self.replay_loader.dataset))

    def _assert_update_schedule_covered(self, logical_steps, update_steps):
        """Catch vector-step regressions that would skip scalar schedule ticks."""
        if not __debug__ or not self.agent_requires_replay:
            return

        update_steps = set(int(step) for step in update_steps)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        schedules = (
            ("update_every_steps", int(getattr(self.agent, "update_every_steps", 1))),
            ("update_actor_every_steps", int(getattr(self.agent, "update_actor_every_steps", 1))),
        )
        special_actor_step = int(getattr(self.agent, "num_expl_steps", 0)) + int(getattr(self.agent, "T_init_steps", 0))

        for name, frequency in schedules:
            assert frequency > 0, f"{name} must be positive, got {frequency}"
            due_steps = []
            for step in logical_steps:
                step = int(step)
                if seed_until_step(step):
                    continue
                if step % frequency == 0 or (name == "update_actor_every_steps" and step == special_actor_step):
                    due_steps.append(step)

            missing = [step for step in due_steps if step not in update_steps]
            assert not missing, (
                f"Parallel stepping would skip {name} due steps {missing}; "
                f"logical_steps={list(map(int, logical_steps))}, update_steps={sorted(update_steps)}"
            )

    def _collect_seed_trajectory_point(self, time_step, logical_step, env_id):
        task_name = str(self.cfg.task_name).lower()
        logical_frame = int(logical_step) * int(self.cfg.action_repeat)
        if ('pointmaze' not in task_name and 'point_maze' not in task_name):
            return
        if logical_frame >= int(self.cfg.num_seed_frames):
            return

        point = np.asarray(
            getattr(time_step, 'proprio_observation', []), dtype=np.float32
        ).reshape(-1)
        if point.size >= 2 and np.all(np.isfinite(point[:2])):
            self._seed_trajectories[env_id].append(point[:2].copy())

        if time_step.last():
            trajectory = self._seed_trajectories[env_id]
            if trajectory:
                self._completed_seed_trajectories.append(
                    np.asarray(trajectory, dtype=np.float32)
                )
            self._seed_trajectories[env_id] = []

    def _maybe_log_seed_trajectories(self):
        task_name = str(self.cfg.task_name).lower()
        is_pointmaze = 'pointmaze' in task_name or 'point_maze' in task_name
        if self._seed_trajectories_logged or not is_pointmaze:
            return
        if self.global_frame < int(self.cfg.num_seed_frames):
            return

        trajectories = list(self._completed_seed_trajectories)
        trajectories.extend(
            np.asarray(trajectory, dtype=np.float32)
            for trajectory in self._seed_trajectories
            if trajectory
        )
        self._seed_trajectories_logged = True
        if not trajectories:
            print('No PointMaze seed trajectory points available to log')
            return

        save_dir = self.work_dir / str(
            getattr(self.cfg, 'seed_trajectory_dir', 'seed_trajectories')
        )
        try:
            paths = save_maze_trajectory_overlay_plot(
                trajectories=trajectories,
                env=self.eval_env,
                step=int(self.cfg.num_seed_frames),
                save_dir=save_dir,
            )
            if self.cfg.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        f'seed_trajectories/{style}': wandb.Image(path)
                        for style, path in paths.items()
                    },
                    step=int(self.cfg.num_seed_frames),
                )
            print(
                f'Logged {len(trajectories)} PointMaze trajectories from '
                f'{int(self.cfg.num_seed_frames)} seed frames'
            )
        except Exception as exc:
            print(f'Could not log PointMaze seed trajectories: {exc}')

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        num_seed_frames = int(self.cfg.num_seed_frames)
        action_repeat = int(self.cfg.action_repeat)
        if num_seed_frames % action_repeat != 0:
            raise ValueError('num_seed_frames must be divisible by action_repeat')
        seed_steps = num_seed_frames // action_repeat
        if seed_steps % self.num_envs != 0:
            raise ValueError(
                'num_seed_frames/action_repeat must be divisible by num_envs so '
                'URL collection ends exactly at the requested frame count'
            )
        collecting = seed_steps > 0
        active_env = self.collection_env if collecting else self.train_env
        observations, infos = active_env.reset(
            seed=[self.base_seed + env_id for env_id in range(self.num_envs)]
        )
        time_steps = self._time_steps_from_infos(infos)
        first_time_step = time_steps[0]
        if self.cfg.obs_type == 'pixels' and hasattr(first_time_step.observation, 'shape'):
            base_channels = 1 if self.cfg.grayscale else 3
            stacked_channels = first_time_step.observation.shape[0]
            effective_frame_stack = stacked_channels // base_channels if base_channels > 0 else 0
            print(
                "Initial observation shape: "
                f"{first_time_step.observation.shape} "
                f"(base_channels={base_channels}, frame_stack={effective_frame_stack})"
            )
        else:
            print(f"Initial observation shape: {first_time_step.observation.shape}")

        metas = [self.agent.init_meta() for _ in range(self.num_envs)]
        episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        for env_id, time_step in enumerate(time_steps):
            self.replay_storage.add(time_step, metas[env_id], env_id=env_id)
        self.train_video_recorder.init(first_time_step.image_observation)
        metrics = None

        while train_until_step(self.global_step):
            logical_steps = self.global_step + np.arange(self.num_envs, dtype=np.int64)

            if any(eval_every_step(int(step)) for step in logical_steps):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            for env_id, step in enumerate(logical_steps):
                metas[env_id] = self.agent.update_meta(
                    metas[env_id],
                    int(step),
                    time_steps[env_id],
                )

            action_steps = logical_steps
            if collecting:
                # Loaded agents often force uniform actions while
                # step < num_expl_steps. Shift only policy-facing steps so URL
                # collection samples from pretrained policy instead. Replay and
                # logging retain true logical steps.
                action_steps = logical_steps + int(
                    getattr(self.agent, 'num_expl_steps', seed_steps)
                )

            with torch.no_grad(), utils.eval_mode(self.agent):
                if hasattr(self.agent, "act_parallel"):
                    actions = self.agent.act_parallel(
                        observations,
                        metas,
                        action_steps,
                        eval_mode=False,
                    )
                else:
                    actions = np.asarray([
                        self.agent.act(
                            observations[env_id],
                            metas[env_id],
                            int(action_steps[env_id]),
                            eval_mode=False,
                        )
                        for env_id in range(self.num_envs)
                    ])

            next_observations, rewards, terminated, truncated, infos = active_env.step(actions)
            done = np.logical_or(terminated, truncated)
            next_time_steps = self._time_steps_from_infos(infos)

            update_steps = []
            for env_id, time_step in enumerate(next_time_steps):
                episode_rewards[env_id] += float(time_step.reward)
                episode_steps[env_id] += 1
                self.replay_storage.add(time_step, metas[env_id], env_id=env_id)
                self._collect_seed_trajectory_point(
                    time_step, logical_steps[env_id], env_id
                )
                if env_id == 0:
                    self.train_video_recorder.record(time_step.image_observation)
                if (self.agent_requires_replay and
                        not seed_until_step(int(logical_steps[env_id]))):
                    update_steps.append(int(logical_steps[env_id]))

            self._assert_update_schedule_covered(logical_steps, update_steps)
            self._drain_encoded_actor_fifo_if_due(logical_steps)
            self._global_step += self.num_envs
            self._maybe_log_seed_trajectories()

            if collecting and self.global_step >= seed_steps:
                for env_id in range(self.num_envs):
                    self.replay_storage.end_episode(env_id)
                observations, infos = self.train_env.reset(
                    seed=[self.base_seed + env_id for env_id in range(self.num_envs)]
                )
                time_steps = self._time_steps_from_infos(infos)
                metas = [self.agent.init_meta() for _ in range(self.num_envs)]
                for env_id, time_step in enumerate(time_steps):
                    self.replay_storage.add(time_step, metas[env_id], env_id=env_id)
                episode_steps.fill(0)
                episode_rewards.fill(0.0)
                active_env = self.train_env
                collecting = False
                self.train_video_recorder.init(time_steps[0].image_observation)
                print(
                    f'Switched from URL collection env to normal train env at '
                    f'{self.global_frame} frames'
                )
                continue

            for logical_step in update_steps:
                metrics = self._update_agent_once(logical_step)

            if np.any(done):
                elapsed_time, total_time = self.timer.reset()
                frames_since_last_log = max(1, self.global_frame - self._last_train_log_frame)
                self._last_train_log_frame = self.global_frame
                train_fps = frames_since_last_log / elapsed_time

                for env_id in np.flatnonzero(done):
                    self._global_episode += 1
                    episode_end_step = int(logical_steps[env_id]) + 1
                    episode_end_frame = episode_end_step * int(self.cfg.action_repeat)
                    if env_id == 0:
                        self.train_video_recorder.save(f'{self.global_frame}.mp4')
                    if metrics is not None or True:
                        episode_frame = int(episode_steps[env_id]) * self.cfg.action_repeat
                        with self.logger.log_and_dump_ctx(episode_end_frame,
                                                        ty='train') as log:
                            log('fps', train_fps)
                            log('total_time', total_time)
                            log('episode_reward', float(episode_rewards[env_id]))
                            log('episode_length', episode_frame)
                            log('episode', self.global_episode)
                            log('buffer_size', len(self.replay_storage))
                            log('step', episode_end_step)
                            self._log_montezuma_episode_metrics(log, next_time_steps[env_id])

                reset_observations, reset_infos = self._reset_done_envs(active_env, done)
                reset_time_steps = self._time_steps_from_infos(reset_infos, required_mask=done)
                observations = reset_observations
                for env_id in range(self.num_envs):
                    if done[env_id]:
                        metas[env_id] = self.agent.init_meta()
                        time_steps[env_id] = reset_time_steps[env_id]
                        self.replay_storage.add(time_steps[env_id], metas[env_id], env_id=env_id)
                        episode_steps[env_id] = 0
                        episode_rewards[env_id] = 0.0
                        if env_id == 0:
                            self.train_video_recorder.init(time_steps[env_id].image_observation)
                    else:
                        time_steps[env_id] = next_time_steps[env_id]
                # self.save_snapshot()
            else:
                observations = next_observations
                time_steps = next_time_steps

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f, map_location=self.device)
        for key, value in payload.items():
            setattr(self, key, value)

    def load_pretrained_snapshot(self):
        snapshot = (
            Path(self.cfg.snapshot_base_dir)
            / self.cfg.obs_type
            / self.cfg.domain
            / self.cfg.agent.name
            / str(self.cfg.seed)
            / f'snapshot_{int(self.cfg.snapshot_ts)}.pt'
        )
        return self.load_snapshot_from_path(snapshot)

    def load_snapshot_from_path(self, path):
        snapshot = Path(path).expanduser()
        print(f'loading pretrained snapshot: {snapshot.resolve()}')
        if not snapshot.exists():
            raise FileNotFoundError(f'Pretrained snapshot not found: {snapshot}')
        with snapshot.open('rb') as stream:
            payload = torch.load(stream, weights_only=False, map_location='cpu')
        if not isinstance(payload, dict) or 'agent' not in payload:
            raise ValueError(f"Snapshot must contain an 'agent' key: {snapshot}")
        return payload

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        if self.global_frame >= self.snapshot_steps[0]:
            snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
            self.snapshot_steps.pop(0)
            print(f'saving snapshot to {snapshot} at frame {self.global_frame}')
        else:
            if self.save_snapshot_flag == False:
                return
            snapshot = snapshot_dir / 'snapshot.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}

        agent = payload['agent']
        restored_refs = []

        def stash_attr(obj, attr, replacement=None):
            # Use __dict__ directly to avoid wrapper __getattr__ recursion while saving.
            obj_dict = getattr(obj, '__dict__', None)
            if obj_dict is None or attr not in obj_dict:
                return
            restored_refs.append((obj, attr, obj_dict[attr]))
            setattr(obj, attr, replacement)

        # Temporarily remove all live environment/debug references before saving.
        # PointMaze/Fetch domain visualizers keep env handles nested under
        # agent.debug_visualizer.domain_visualizer; those env wrappers are not
        # safely pickleable and can recurse during torch.load.
        stash_attr(agent, 'env')
        stash_attr(agent, 'wrapped_env')
        stash_attr(agent, '_discrete_env')
        stash_attr(agent, 'visualizer')
        stash_attr(agent, 'gridworld_visualizer')
        stash_attr(agent, 'domain_visualizer')

        debug_visualizer = getattr(agent, '__dict__', {}).get('debug_visualizer', None)
        stash_attr(debug_visualizer, 'domain_visualizer')

        try:
            with snapshot.open('wb') as f:
                torch.save(payload, f)
        finally:
            for obj, attr, value in reversed(restored_refs):
                setattr(obj, attr, value)

    def close(self):
        replay_iter = getattr(self, "_replay_iter", None)
        shutdown_workers = getattr(replay_iter, "_shutdown_workers", None)
        if callable(shutdown_workers):
            try:
                shutdown_workers()
            except Exception as exc:
                print(f"Could not shut down replay workers cleanly: {exc}")

        for recorder_name in ("video_recorder", "train_video_recorder"):
            recorder = getattr(self, recorder_name, None)
            if recorder is not None and hasattr(recorder, "frames"):
                recorder.frames = []

        for env_name in ("eval_env", "train_env", "collection_env"):
            env = getattr(self, env_name, None)
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    print(f"Could not close {env_name} cleanly: {exc}")


@hydra.main(config_path='configs', config_name='train_parallel/finetune_maze', version_base='1.1')
def main(cfg):
    W = Workspace
    root_dir = Path.cwd()
    if not hasattr(cfg, 'save_log'):
        with open_dict(cfg):
            cfg.save_log = True

    log_context = enable_console_log(root_dir / 'train_parallel.log') if cfg.save_log else NullContext()
    with log_context:
        workspace = W(cfg)
        try:
            snapshot = root_dir / 'snapshot.pt'
            if snapshot.exists():
                print(f'resuming: {snapshot}')
                workspace.load_snapshot()
            workspace.train()
        finally:
            workspace.close()


if __name__ == '__main__':
    main()
