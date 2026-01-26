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
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

# Registra un resolver personalizzato solo se non esiste già
if not OmegaConf.has_resolver("get_filename"):
    OmegaConf.register_new_resolver(
        "get_filename",
        lambda path: os.path.splitext(os.path.basename(path))[0] if path != "none" else "none"
    )

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
        if cfg.seed == 1:
            cfg.seed = np.random.randint(1, 10000)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        
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
                    mode=cfg.wandb_mode if hasattr(cfg, 'wandb_mode') else 'online')
            else:
                wandb.init(
                    config=OmegaConf.to_container(cfg, resolve=True),
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=True,
                    mode=cfg.wandb_mode if hasattr(cfg, 'wandb_mode') else 'online')


        # create envs
        task = cfg.task_name
        if hasattr(cfg, 'env'):
            env_kwargs = gym_env.make_kwargs(cfg)
        else:
            env_kwargs = {}
        self.train_env = gym_env.make(self.cfg.task_name, self.cfg.obs_type, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal, url=False, **env_kwargs)
        self.collection_env = gym_env.make(self.cfg.task_name, self.cfg.obs_type, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal, url=True, **env_kwargs)
        
        self.eval_env = gym_env.make(self.cfg.task_name, self.cfg.obs_type, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal, url=False, **env_kwargs)
       
        # Get observation and action specs for the agent
        obs_spec = gym_env.observation_spec(self.collection_env)
        action_spec = gym_env.action_spec(self.collection_env)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                obs_spec,
                                action_spec,
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)
        

        # TODO Remove
        self.INITIAL_HEATMAP = False
        self.dataset = {
            'states': np.array([]),
            'actions': np.array([]),
            'rewards': np.array([]),
        }

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)
            # Re-insert environment after loading
            if hasattr(self.agent, 'insert_env'):
                self.agent.insert_env(self.train_env)
        
        if cfg.p_path is not None and cfg.p_path != "none":
            if cfg.p_path.endswith(".npy"):
                self.agent = utils.load_policy_weights_into_agent(self.agent, cfg.p_path, device=self.device)
            else:
                pretrained_agent = self.load_snapshot_from_path(cfg.p_path)['agent']
                self.agent.init_from(pretrained_agent)
                # Re-insert environment after loading
                if hasattr(self.agent, 'insert_env'):
                    self.agent.insert_env(self.train_env)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()

        # create replay buffer
        data_specs = (obs_spec,
                      action_spec,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

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
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            use_wandb=self.cfg.use_wandb,
            is_training_sample=False)

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
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
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
        if not seed_until_step(self.global_step):
            time_step = self.train_env.reset()
        else:
            time_step = self.collection_env.reset()
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.image_observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                # if metrics is not None:
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

                # reset env
                if not seed_until_step(self.global_step):
                    time_step = self.train_env.reset()
                else:
                    time_step = self.collection_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.image_observation)

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            if hasattr(self.agent, "regress_meta"):
                repeat = self.cfg.action_repeat
                every = self.agent.update_task_every_step // repeat
                init_step = self.agent.num_init_steps
                if self.global_step > (
                        init_step // repeat) and self.global_step % every == 0:
                    meta = self.agent.regress_meta(self.replay_iter,
                                                   self.global_step)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            if self.global_step > self.cfg.num_seed_frames + (self.cfg.agent.update_actor_after_critic_steps if hasattr(self.cfg.agent, "update_actor_after_critic_steps") else self.cfg.update_actor_after_critic_steps):
                if not self.INITIAL_HEATMAP:
                        self.visualize_dataset_heatmap("dataset_heatmap.png")
                        self.INITIAL_HEATMAP = True
            # try to update the agent
            if not seed_until_step(self.global_step):
                for _ in range(self.cfg.num_agent_updates_per_env_step):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            if not seed_until_step(self.global_step):
                time_step = self.train_env.step(action)
            else:
                time_step = self.collection_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            if not self.INITIAL_HEATMAP:
                self.dataset['states'] = np.append(self.dataset['states'],  time_step.proprio_observation if time_step.proprio_observation.shape[0] == 2 else  np.argmax(time_step.proprio_observation))
                self.dataset['actions'] = np.append(self.dataset['actions'], time_step.action)
                self.dataset['rewards'] = np.append(self.dataset['rewards'], time_step.reward)
            self.train_video_recorder.record(time_step.image_observation)
            episode_step += 1
            self._global_step += 1

    def visualize_dataset_heatmap(self, save_path: str) -> None:
        """
        Visualize dataset state visitation as heatmap.
        
        Args:
            save_path: Path to save the heatmap
        """
        # Check if environment has cells attribute (grid-based environment)
        if hasattr(self.train_env.unwrapped, 'cells'):
            # Get grid dimensions
            max_x = max(cell[0] for cell in self.train_env.unwrapped.cells)
            max_y = max(cell[1] for cell in self.train_env.unwrapped.cells)
            min_x = min(cell[0] for cell in self.train_env.unwrapped.cells)
            min_y = min(cell[1] for cell in self.train_env.unwrapped.cells)
            grid_width = max_x - min_x + 1
            grid_height = max_y - min_y + 1
            
            # Count state visitations
            state_counts = np.zeros(self.train_env.unwrapped.n_states)
            for state in self.dataset['states']:
                state_counts[int(state)] += 1
            
            # Create grid
            grid = np.zeros((grid_height, grid_width))
            for s_idx in range(self.train_env.unwrapped.n_states):
                x, y = self.train_env.unwrapped.idx_to_state[s_idx]
                grid[y - min_y, x - min_x] = state_counts[s_idx]
            
            # Mask zero values to show background color
            masked_grid = np.ma.masked_where(grid == 0, grid)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            # Set light gray background
            ax.set_facecolor("#B2B2B2")
            # Plot only non-zero values
            im = ax.imshow(masked_grid, cmap='YlOrRd', interpolation='nearest')
            ax.set_title(f'Dataset State Visitation (n={len(self.dataset["states"])})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
            plt.colorbar(im, ax=ax, label='Visit Count')
        else:
            # Continuous space: plot (x,y) coordinates as scatter plot
            states = self.dataset['states']
            if len(states) == 0:
                return
            
            # Reshape states to extract x,y coordinates
            states = np.array(states).reshape(-1, 2)
            x_coords = states[:, 0]
            y_coords = states[:, 1]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            # Create scatter plot
            ax.scatter(x_coords, y_coords, c='red', alpha=0.5, s=10)
            ax.set_title(f'Dataset State Visitation (n={len(states)})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain = self.cfg.domain
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            
            print(f'trying to load snapshot from absolute path: {os.path.abspath(snapshot)}')
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        # while True:
        #     seed = np.random.randint(1, 11)
        #     payload = try_load(seed)
        #     if payload is not None:
        #         return payload
        return None

    def load_snapshot_from_path(self, path: str):
        snapshot = Path(path)
        print(f'loading snapshot from path: {os.path.abspath(snapshot)}')
        if not snapshot.exists():
            return None
        with snapshot.open('rb') as f:
            payload = torch.load(f, weights_only=False, map_location='cpu')
            print(f"Loaded snapshot keys: {list(payload.keys())}")
        return payload


@hydra.main(config_path='.', config_name='train')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
