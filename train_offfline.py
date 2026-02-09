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
from dm_env import specs
import gym_env
import wandb

import utils
from logger import Logger
from replay_buffer import make_offline_replay_loader
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith('point_mass_maze'):
        return 'point_mass_maze'
    return task.split('_', 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1

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

def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    while eval_until_episode(episode):
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation,
                                   global_step,
                                   eval_mode=True)
            time_step = env.step(action)
            video_recorder.record(env)
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f'{global_step}.mp4')

    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('episode_reward', total_reward / episode)
        log('episode_length', step / episode)
        log('step', global_step)


@hydra.main(config_path='.', config_name='config')
def main(cfg):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb)

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
                
    # create envs
    task = cfg.task_name
    if hasattr(cfg, 'env'):
        env_kwargs = gym_env.make_kwargs(cfg)
    else:
        env_kwargs = {}

    env = gym_env.make(cfg.task_name, cfg.obs_type, cfg.frame_stack,
                    cfg.action_repeat, cfg.seed, cfg.resolution, cfg.random_init, 
                    cfg.random_goal, url=True, **env_kwargs)
    

    # create agent
    agent = make_agent(cfg.obs_type,
                        obs_spec,
                        action_spec,
                        0, 
                        cfg.agent)
    

    # create replay buffer
    # Get observation and action specs for the agent
    obs_spec = gym_env.observation_spec(env)
    action_spec = gym_env.action_spec(env)

    # get meta spec
    meta_specs = agent.get_meta_specs()
    # create replay buffer
    data_specs = (obs_spec,
                    action_spec,
                    specs.Array((1,), np.float32, 'reward'),
                    specs.Array((1,), np.float32, 'discount'))

    # create data storage
    domain = get_domain(cfg.task)
    # datasets_dir = work_dir / cfg.replay_buffer_dir
    # replay_dir = datasets_dir.resolve() / domain / cfg.expl_agent / 'buffer'
    replay_dir = cfg.replay_buffer_dir
    print(f'replay dir: {replay_dir}')

    replay_loader = make_offline_replay_loader(env, replay_dir, cfg.replay_buffer_size,
                                       cfg.batch_size,
                                       cfg.replay_buffer_num_workers,
                                       False,
                                       cfg.step,
                                       cfg.discount)
    replay_iter = iter(replay_loader)

    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()

    global_step = 0

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    while train_until_step(global_step):
        # try to evaluate
        if eval_every_step(global_step):
            logger.log('eval_total_time', timer.total_time(), global_step)
            eval(global_step, agent, env, logger, cfg.num_eval_episodes,
                 video_recorder)

        metrics = agent.update(replay_iter, global_step)
        logger.log_metrics(metrics, global_step, ty='train')
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty='train') as log:
                log('fps', cfg.log_every_steps / elapsed_time)
                log('total_time', total_time)
                log('step', global_step)

        global_step += 1


if __name__ == '__main__':
    main()