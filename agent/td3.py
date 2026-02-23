import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils

class Encoder(nn.Module):
    def __init__(self, obs_shape, obs_type='pixels'):
        super().__init__()
        
        self.obs_type = obs_type
        
        if obs_type == 'pixels':
            assert len(obs_shape) == 3
            self.repr_dim = 32 * 35 * 35

            self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                         nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                         nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                         nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                         nn.ReLU())
        elif obs_type in ['states', 'discrete_states']:
            self.repr_dim = obs_shape[0]
            self.convnet = nn.Identity()
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

        self.apply(utils.weight_init)

    def forward(self, obs):
        if self.obs_type == 'pixels':
            obs = obs / 255.0 - 0.5
            h = self.convnet(obs)
            h = h.view(h.shape[0], -1)
            return h
        else:
            return obs
        
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)

        return q1, q2


class TD3Agent:
    def __init__(self,
                 name,
                 obs_shape,
                 obs_type,
                 action_shape,
                 device,
                 lr,
                 hidden_dim,
                 critic_target_tau,
                 epsilon_schedule,
                 nstep,
                 batch_size,
                 use_tb,
                 has_next_action=False,
                 meta_dim=0):
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.epsilon_schedule = epsilon_schedule
        raise NotImplementedError("Bisogna modificare l'input del critic per gestire la rappresentazione, mettere aug and encode su pixels, farlo per azioni discrete ma non so se a questo punto usare direttamente ddpg...")

        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape, obs_type='pixels').to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        elif obs_type in ['states', 'discrete_states']:
            self.aug = nn.Identity()
            self.encoder = Encoder(obs_shape, obs_type=obs_type).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")
        
        # models
        self.actor = Actor(self.obs_dim, action_shape[0],
                           hidden_dim).to(device)

        self.critic = Critic(self.obs_dim, action_shape[0],
                             hidden_dim).to(device)
        self.critic_target = Critic(self.obs_dim, action_shape[0],
                                    hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.encoder.train(training)
        self.critic.train(training)
    
    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, stddev)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, stddev)

        Q1, Q2 = self.critic(obs, policy.sample(clip=self.stddev_clip))
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, action, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics