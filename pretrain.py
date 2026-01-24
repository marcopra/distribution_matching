import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

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
        task = cfg.task_name
        if hasattr(cfg, 'env'):
            env_kwargs = gym_env.make_kwargs(cfg)
        else:
            env_kwargs = {}
        self.train_env = gym_env.make(self.cfg.task_name, self.cfg.obs_type, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal, url=True, **env_kwargs)
        self.eval_env = gym_env.make(self.cfg.task_name, self.cfg.obs_type, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal, url=True, **env_kwargs)
       

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
            # Pass the wrapped train_env, not unwrapped
            self.agent.insert_env(self.train_env)
    

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
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
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
            is_training_sample=False)
        
        self.snapshot_steps = cfg.snapshots

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

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

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        eval_mode = False
        if eval_mode == False:
            utils.ColorPrint.yellow("Evaluating with eval_mode=False")
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=eval_mode) # I am not sure we should evaluate with eval_mode=True during pretrain... ORIGINAL CODE: True
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
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.image_observation)
        metrics = None
        while train_until_step(self.global_step):
            print(f"Starting training step {self.global_step}", end='\r')
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
            snapshot = snapshot_dir / 'snapshot.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        
        # Temporarily remove all environment references before saving
        env_ref = None
        wrapped_env_ref = None
        discrete_env_ref = None
        visualizer_ref = None
        
        if hasattr(payload['agent'], 'env'):
            env_ref = payload['agent'].env
            payload['agent'].env = None
        if hasattr(payload['agent'], 'wrapped_env'):
            wrapped_env_ref = payload['agent'].wrapped_env
            payload['agent'].wrapped_env = None
        if hasattr(payload['agent'], '_discrete_env'):
            discrete_env_ref = payload['agent']._discrete_env
            payload['agent']._discrete_env = None
        if hasattr(payload['agent'], 'visualizer'):
            visualizer_ref = payload['agent'].visualizer
            payload['agent'].visualizer = None
            
        with snapshot.open('wb') as f:
            torch.save(payload, f)
        
        # Restore all references after saving
        if env_ref is not None:
            payload['agent'].env = env_ref
        if wrapped_env_ref is not None:
            payload['agent'].wrapped_env = wrapped_env_ref
        if discrete_env_ref is not None:
            payload['agent']._discrete_env = discrete_env_ref
        if visualizer_ref is not None:
            payload['agent'].visualizer = visualizer_ref


@hydra.main(config_path='.', config_name='pretrain')
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
