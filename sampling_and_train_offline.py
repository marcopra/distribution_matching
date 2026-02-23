import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import matplotlib.pyplot as plt
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
from copy import deepcopy

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

        self.train_env = gym_env.make(self.cfg.task_name, self.cfg.obs_type, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal, url=True, **env_kwargs)
        self.eval_env = gym_env.make(self.cfg.task_name, self.cfg.obs_type, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal, url=False, **env_kwargs)
       
        # TODO Remove
        self.dataset = {
            'states': np.array([]),
            'actions': np.array([]),
            'rewards': np.array([]),
        }
        # Get observation and action specs for the agent
        obs_spec = gym_env.observation_spec(self.train_env)
        action_spec = gym_env.action_spec(self.train_env)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                obs_spec,
                                action_spec,
                                None, #cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)
        
        self.initial_agent = deepcopy(self.agent)  # make a copy of the initial agent

        # initialize from pretrained
        # if cfg.pretrained_path is not None and cfg.pretrained_path != "none":
        #     if cfg.pretrained_path.endswith('.pt'):
        #         self.agent = self.load_sampler(cfg.pretrained_path)['agent']
        #     elif cfg.pretrained_path.endswith('.npy'):
        #         self.agent = DistMatchingAgent(env=self.train_env)
        #         self.agent.load_policy_operator(cfg.pretrained_path)
        #     print(f'Loaded pretrained agent {type(self.agent)} from: {cfg.pretrained_path}')  
        # else:
        #     self.agent = deepcopy(self.agent)
        #     print(f'No pretrained agent specified, using training agent as sampling agent.')
            
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
                                                True, cfg.nstep, cfg.discount)
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
        self._sampling_step = 0
        self._training_step = 0
        self._total_training_steps = 0
        self._global_step = 0
        self._global_episode = 0
        self._agent_updates = 0
        self.current_cycle = 0
        self.cfg.num_seed_frames = cfg.num_seed_frames_array[0]

    @property
    def global_step(self):
        return self._global_step

    @property
    def training_step(self):
        return self._training_step

    @property
    def sampling_step(self):
        return self._sampling_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.sampling_step * self.cfg.action_repeat

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

        # self.visualize_q_values_bars(f"q_values_bars_training_step_{self._training_step}.png")

    # def eval_dynamic_programming(self):
    #     step, episode, total_reward = 0, 0, 0
    #     eval_until_episode = utils.Until(self.cfg.num_eval_dp_episodes)
    #     meta = self.agent.init_meta()
    #     while eval_until_episode(episode):
    #         episode_reward = 0
    #         time_step = self.eval_env.reset()
    #         self.video_recorder.init(self.eval_env, enabled=(episode == 0))
    #         while not time_step.last():
    #             with torch.no_grad(), utils.eval_mode(self.agent):
    #                 action = self.agent.act(time_step.observation,
    #                                         meta,
    #                                         1000000000, # self.global_step Need o reduce eps greedyness during DP eval id we use eval_mode=False
    #                                         eval_mode=False) # TODO think more if to use deterministic or stochastic
    #             time_step = self.eval_env.step(action)
    #             self.video_recorder.record(self.eval_env)
    #             episode_reward += time_step.reward
    #             total_reward += time_step.reward
    #             step += 1
    #         print("reward for eval episode {}: {}".format(episode, episode_reward))
    #         episode += 1
    #         self.video_recorder.save(f'{self.global_frame}.mp4')

    #     with self.logger.log_and_dump_ctx(self.global_frame, ty='eval_dp') as log:
    #         log('episode_reward', total_reward / episode)
    #         print("Average reward over eval episodes: {}".format(total_reward / episode))
    #         log('episode_length', step * self.cfg.action_repeat / episode)
    #         log('episode', self.global_episode)
    #         log('dataset_size', len(self.replay_storage))
    #         log('dynamic_programming_cycle', self.current_cycle)
    #         log('step', self.global_step)
    #         log('total_time', self.timer.total_time())

    def collect_and_train(self):
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
        # Data collection loop
        while seed_until_step(self._sampling_step):
            if time_step.last():
                self._global_episode += 1
                print("Sampled episode:", self.global_episode, " and step:", self._sampling_step)
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
               
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
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                # self.dataset['states'] = np.append(self.dataset['states'], np.argmax(time_step.observation))
                # self.dataset['actions'] = np.append(self.dataset['actions'], time_step.action)
                # self.dataset['rewards'] = np.append(self.dataset['rewards'], time_step.reward)
                self.train_video_recorder.init(time_step.image_observation)

                episode_step = 0
                episode_reward = 0

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
                                        self.sampling_step,
                                        eval_mode=False)
            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            # self.dataset['states'] = np.append(self.dataset['states'], np.argmax(time_step.observation))
            # self.dataset['actions'] = np.append(self.dataset['actions'], time_step.action)
            # self.dataset['rewards'] = np.append(self.dataset['rewards'], time_step.reward)
            self.train_video_recorder.record(time_step.image_observation)
            episode_step += 1
            self._sampling_step += 1
        
        # Training loop
        all_losses = {}  # accumula tutte le loss metrics
        while train_until_step(self.training_step):
            # evaluation
            if eval_every_step(self._training_step):
                print(f'\nTraining step {self._training_step}, starting evaluation:')
                self.eval()

            # try to update the agent
            metrics = self.agent.update(self.replay_iter, self.training_step)
            self.logger.log_metrics(metrics, self._total_training_steps, ty='train')
            if self.cfg.use_wandb:
                wandb_data = {'train' + '/' + key: val for key, val in metrics.items()}
                wandb_data['train/total_train_step'] = self._total_training_steps
                wandb_data['train/relative_train_step'] = self._training_step
                wandb.log(wandb_data)

            # accumula le loss per il summary finale
            for key, value in metrics.items():
                if 'loss' in key.lower():
                    if key not in all_losses:
                        all_losses[key] = []
                    all_losses[key].append(value)
            
            self._global_step += 1
            self._training_step += 1
            self._total_training_steps += 1
        
        # stampa summary delle loss alla fine del training
        print('\n' + '='*80)
        print('Training Summary - Losses')
        print('='*80)
        for key, values in sorted(all_losses.items()):
            mean_val = np.mean(values)
            print(f'  {key}: initial {values[0]:.6f} ended {values[-1]:.6f}  min={np.min(values):.6f}, max={np.max(values):.6f}')
        print('='*80 + '\n')

    def dynamic_programming_loop(self):

        while self.current_cycle < len(self.cfg.num_seed_frames_array):
            print(f'\nStarting dynamic programming cycle {self.current_cycle} with {self.cfg.num_seed_frames} seed frames.\n')
             # set the number of seed frames for this cycle
            self.cfg.num_seed_frames = self.cfg.num_seed_frames_array[self.current_cycle]
            self.collect_and_train()
            self.eval_dynamic_programming()
            self.visualize_dataset_heatmap(f"heatmap_cycle_{self.current_cycle}.png")
            self.visualize_q_values_bars(f"q_values_bars_cycle_{self.current_cycle}.png")
            self.reset_for_next_cycle()
            self.current_cycle += 1
    
    def reset_for_next_cycle(self):
        self._training_step = 0
        self.agent = deepcopy(self.initial_agent)
        if not (self.cfg.pretrained_path is not None and self.cfg.pretrained_path != "none"):
            self.agent = deepcopy(self.agent)
        self._replay_iter = None  # reset replay iterator

    def visualize_dataset_heatmap(self, save_path: str) -> None:
        """
        Visualize dataset state visitation as heatmap.
        
        Args:
            save_path: Path to save the heatmap
        """
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
        for s_idx in range( self.train_env.unwrapped.n_states):
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
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def visualize_q_values_bars(self, save_path: str) -> None:
        """
        Visualize Q-values as action probability bars for each state.
        
        Args:
            save_path: Path to save the visualization
        """
        from matplotlib.patches import Rectangle, Patch
        
        # Get grid dimensions
        max_x = max(cell[0] for cell in self.train_env.unwrapped.cells)
        max_y = max(cell[1] for cell in self.train_env.unwrapped.cells)
        min_x = min(cell[0] for cell in self.train_env.unwrapped.cells)
        min_y = min(cell[1] for cell in self.train_env.unwrapped.cells)
        grid_width = max_x - min_x + 1
        grid_height = max_y - min_y + 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-0.5, grid_width - 0.5)
        ax.set_ylim(grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Q-Values Action Probabilities per State (Cycle {self.current_cycle})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        action_colors = ['red', 'blue', 'green', 'orange']
        action_names = ['up', 'down', 'left', 'right']
        n_actions = self.train_env.unwrapped.action_space.n
        
        # Get Q-values for all states
        with torch.no_grad(), utils.eval_mode(self.agent):
            for s_idx in range(self.train_env.unwrapped.n_states):
                x, y = self.train_env.unwrapped.idx_to_state[s_idx]
                x_plot, y_plot = x - min_x, y - min_y
                
                # Draw background
                rect = Rectangle((x_plot - 0.4, y_plot - 0.4), 0.8, 0.8,
                               facecolor='lightgray', edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                
                # Create one-hot observation for this state
                obs = np.zeros(self.train_env.unwrapped.n_states, dtype=np.float32)
                obs[s_idx] = 1.0
                obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                
                # Get Q-values from agent
                h = self.agent.encoder(obs_tensor)
                meta = self.agent.init_meta()
                inputs = [h]
                for value in meta.values():
                    value = torch.as_tensor(value, device=self.device).unsqueeze(0)
                    inputs.append(value)
                inpt = torch.cat(inputs, dim=-1)
                
                q1, q2 = self.agent.critic(inpt)
                q_values = torch.min(q1, q2).squeeze(0).cpu().numpy()
                
                # Convert Q-values to probabilities using softmax
                probs = np.exp(q_values - np.max(q_values))  # numerical stability
                probs = probs / probs.sum()
                
                # Draw mini bar chart
                bar_width = 0.15
                bar_spacing = 0.2
                start_x = x_plot - 1.5 * bar_spacing
                max_bar_height = 0.7
                
                for a_idx in range(n_actions):
                    bar_x = start_x + a_idx * bar_spacing
                    bar_height = probs[a_idx] * max_bar_height
                    
                    bar_rect = Rectangle((bar_x - bar_width/2, y_plot + 0.35 - bar_height),
                                        bar_width, bar_height,
                                        facecolor=action_colors[a_idx],
                                        edgecolor='black', linewidth=0.3)
                    ax.add_patch(bar_rect)
        
        # Add legend
        legend_elements = [Patch(facecolor=action_colors[i], label=action_names[i])
                          for i in range(n_actions)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
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
                print(payload.keys())
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

    def load_sampler(self, pretrained_path): # TODO modify
        with open(pretrained_path, 'rb') as f:
                payload = torch.load(f, weights_only=False)
                print(payload.keys())
        return payload
        # snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        # domain = self.cfg.domain
        # snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        # def try_load(seed):
        #     snapshot = snapshot_dir / str(
        #         seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            
        #     print(f'trying to load snapshot from absolute path: {os.path.abspath(snapshot)}')
        #     if not snapshot.exists():
        #         return None
        #     with open(pretrained_path, 'rb') as f:
        #         payload = torch.load(f)
        #         print(payload.keys())
        #     return payload

        # # try to load current seed
        # payload = try_load(self.cfg.seed)
        # if payload is not None:
        #     return payload
        # # otherwise try random seed
        # while True:
        #     seed = np.random.randint(1, 11)
        #     payload = try_load(seed)
        #     if payload is not None:
        #         return payload
        # return None


@hydra.main(config_path='configs', config_name='train/sampling_and_train_offline', version_base='1.1')
def main(cfg):
    from sampling_and_train_offline import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    # workspace.dynamic_programming_loop()
    workspace.collect_and_train()


if __name__ == '__main__':
    main()
