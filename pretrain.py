import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

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
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import ale_py
from omegaconf import open_dict
from agent.utils_debug_visualization import (
    extract_eval_trajectory_point,
    save_eval_trajectory_plots,
)


torch.backends.cudnn.benchmark = True


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
                    sync_tensorboard=True,
                    mode='online')
            else:
                wandb.init(
                    config=OmegaConf.to_container(cfg, resolve=True),
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=True,
                    mode='online')
                
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        env_kwargs = OmegaConf.to_container(cfg.env, resolve=True) if hasattr(cfg, 'env') else {}
        env_kwargs.pop('name', None)
        env_kwargs.pop('synthetic_first_transition', None)

        self.train_env = gym_env.make(
            self.cfg.task_name,
            self.cfg.obs_type,
            frame_stack=self.cfg.frame_stack,
            action_repeat=self.cfg.action_repeat,
            seed=self.cfg.seed,
            resolution=self.cfg.resolution,
            grayscale=self.cfg.grayscale,
            url=True,
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
            url=True,
            **env_kwargs,
        )
       
        # TODO: modify the make function to work with cfg and modify inplace the cfg values, this is a temporary solution to avoid modifying the make function
        if isinstance(self.train_env.unwrapped, ale_py.env.AtariEnv) or str(self.cfg.task_name).startswith("ALE/"):
            # L'action repeat è gestito internamente da ALE, quindi forziamo action_repeat a 1
            with open_dict(self.cfg):
                self.cfg.action_repeat = 1
        # Get observation and action specs for the agent
        obs_spec = gym_env.observation_spec(self.train_env)
        action_spec = gym_env.action_spec(self.train_env)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                obs_spec,
                                action_spec,
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        time_step = self.train_env.reset()

    
        if hasattr(self.agent, 'insert_env'):
            # Use eval_env for debug rollouts so visualization does not disturb training.
            self.agent.insert_env(self.eval_env)
    

        # create replay buffer
        data_specs = (obs_spec,
                      action_spec,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      )

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        first_transition = type(self.agent).__name__ == 'RoverAgent'
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                cfg.save_buffer, cfg.nstep, cfg.discount,
                                                first_transition=first_transition)
        
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

    @property
    def is_montezuma(self):
        return 'MontezumaRevenge' in str(self.cfg.task_name)

    def _get_time_step_info(self, time_step):
        info = getattr(time_step, 'info', None)
        return info if isinstance(info, dict) else {}

    def _log_montezuma_episode_metrics(self, log, time_step):
        if not self.is_montezuma:
            return
        info = self._get_time_step_info(time_step)
        if 'montezuma_visited_second_room' in info:
            log('montezuma_visited_second_room',
                float(info['montezuma_visited_second_room']))
        if 'montezuma_max_room_id' in info and info['montezuma_max_room_id'] is not None:
            log('montezuma_max_room_id', info['montezuma_max_room_id'])

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
        eval_trajectories = []
        eval_mode = False
        if eval_mode == False:
            utils.ColorPrint.yellow("Evaluating with eval_mode=False")
        while eval_until_episode(episode):
            meta = self.agent.init_meta()
            time_step = self.eval_env.reset()
            trajectory = []
            point = extract_eval_trajectory_point(self.eval_env, time_step)
            if point is not None:
                trajectory.append(point)
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=eval_mode) # I am not sure we should evaluate with eval_mode=True during pretrain... ORIGINAL CODE: True
                time_step = self.eval_env.step(action)
                point = extract_eval_trajectory_point(self.eval_env, time_step)
                if point is not None:
                    trajectory.append(point)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            if trajectory:
                eval_trajectories.append(trajectory)
            self.video_recorder.save(f'{self.global_frame}.mp4')

        self._save_eval_trajectory_plots(eval_trajectories)

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if self.is_montezuma and episode > 0:
                info = self._get_time_step_info(time_step)
                if 'montezuma_visited_second_room' in info:
                    log('montezuma_visited_second_room',
                        float(info['montezuma_visited_second_room']))
                if 'montezuma_max_room_id' in info and info['montezuma_max_room_id'] is not None:
                    log('montezuma_max_room_id', info['montezuma_max_room_id'])

    def _save_eval_trajectory_plots(self, trajectories):
        enabled = getattr(self.cfg, "plot_eval_trajectories", False)
        if not enabled or not trajectories:
            return

        save_dir = getattr(self.cfg, "eval_trajectory_plot_dir", "eval_trajectory_plots")
        save_dir = self.work_dir / Path(save_dir)
        styles = getattr(self.cfg, "eval_trajectory_plot_styles", None)
        if styles is not None:
            styles = tuple(styles)

        for checkpoint in self._eval_trajectory_plot_checkpoints(len(trajectories)):
            checkpoint_trajectories = trajectories[:checkpoint]
            try:
                save_eval_trajectory_plots(
                    trajectories=checkpoint_trajectories,
                    env=self.eval_env,
                    step=self.global_frame,
                    save_dir=save_dir,
                    styles=styles,
                )
            except Exception as exc:
                print(f"⚠ Could not generate evaluation trajectory plots: {exc}")

    def _eval_trajectory_plot_checkpoints(self, n_trajectories):
        plot_episodes = getattr(self.cfg, "eval_trajectory_plot_episodes", None)
        if plot_episodes is None or len(plot_episodes) == 0:
            return [n_trajectories]

        checkpoints = {int(episode) for episode in plot_episodes}
        checkpoints.add(n_trajectories)
        return [
            checkpoint
            for checkpoint in sorted(checkpoints)
            if 1 <= checkpoint <= n_trajectories
        ]

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        if self.cfg.obs_type == 'pixels' and hasattr(time_step.observation, 'shape'):
            base_channels = 1 if self.cfg.grayscale else 3
            stacked_channels = time_step.observation.shape[0]
            effective_frame_stack = stacked_channels // base_channels if base_channels > 0 else 0
            print(
                "Initial observation shape: "
                f"{time_step.observation.shape} "
                f"(base_channels={base_channels}, frame_stack={effective_frame_stack})"
            )
        else:
            print(f"Initial observation shape: {time_step.observation.shape}")
        meta = self.agent.init_meta()
        self._maybe_set_synthetic_first_transition(time_step, meta)
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.image_observation)
        metrics = None
        while train_until_step(self.global_step):
            # print(f"Starting training step {self.global_step}", end='\r')
            # if time_step.last() or (hasattr(self.agent, "dataset") and self.agent.dataset.reset_episode):
            if time_step.last() or (hasattr(self.agent, "dataset") and self.agent.dataset.reset_episode):
                self._global_episode += 1
                
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                    ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        self._log_montezuma_episode_metrics(log, time_step)

                if type(self.agent).__name__ == "DistMatchingEmbeddingAgent":
                    meta = self.agent.update_meta(meta, self.global_step, time_step)
                    
                # reset env
                time_step = self.train_env.reset()

                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.image_observation)
                # try to save snapshot
                self.save_snapshot()

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                update_signature = inspect.signature(self.agent.update)
                if 'replay_buffer' in update_signature.parameters:
                    metrics = self.agent.update(
                        self.replay_iter,
                        self.global_step,
                        replay_buffer=self.replay_loader.dataset,
                    )
                else:
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.image_observation)
            episode_step += 1
            self._global_step += 1

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


@hydra.main(config_path='configs', config_name='pretrain/pretrain_atari', version_base='1.1')
def main(cfg):
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
