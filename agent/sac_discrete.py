import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
from agent import Agent
from utils import ColorPrint
from collections import OrderedDict

import utils

import hydra

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, linear= False):
        super().__init__()

        if linear:
            self.trunk = nn.Identity(gradient=False)
            self.policy = nn.Linear(obs_dim, action_dim)
            self.apply(utils.weight_init)
            ColorPrint.yellow("Using linear actor!")
            return

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)
        

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)

        logits = self.policy(h)
       
        return F.softmax(logits, dim=1)



class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, action_dim)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs):
        inpt = obs
        h = self.trunk(inpt)

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2
    

class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self,  
                 name,
                 obs_shape,
                 obs_type,
                 action_shape, # Number of discrete actions
                 feature_dim,
                 hidden_dim,
                 device, 
                 use_tb,
                 use_wandb,
                 num_expl_steps,
                 update_every_steps,
                 nstep,
                 init_temperature, 
                 alpha_lr, 
                 actor_lr, 
                 actor_update_frequency, 
                 critic_lr,
                 critic_target_tau, 
                 critic_target_update_frequency,
                 batch_size, 
                 learnable_temperature,
                 linear_actor=False,
                 meta_dim=0):
        super().__init__()

        self.device = torch.device(device)
        self.critic_target_tau = critic_target_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.action_dim = action_shape[0]
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.meta_dim = meta_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.update_every_steps = update_every_steps

        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim

        self.actor = Actor(obs_type, self.obs_dim, self.action_dim,
                           feature_dim, hidden_dim, linear=linear_actor).to(device)

        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -self.action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    
    def act(self, obs, meta, step, eval_mode):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(obs)

            if eval_mode:
                action = probs.argmax(dim=-1).cpu().numpy()[0]
            else:
                action = Categorical(probs).sample().item()
                if step < self.num_expl_steps:
                    # sample the discrete action uniformly during initial exploration
                    action = np.random.randint(self.action_dim)
        return action
    

    def update_critic(self, obs, action, reward, next_obs, discount,
                      step):
        metrics = dict()    
        with torch.no_grad():   
            next_probs = self.actor(next_obs)
            next_log_probs = torch.log(next_probs + 1e-8)
            target_Q1, target_Q2 = self.critic_target(next_obs)    
            target_V = torch.sum(next_probs * (torch.min(target_Q1,target_Q2) - self.alpha.detach() * next_log_probs), dim=1, keepdim=True) # [B,1]
            target_Q = reward + ((discount!=0.0)*0.99 * target_V) # termination is in the discount

        # get current Q estimates
        Q1_all, Q2_all = self.critic(obs)
        current_q1, current_q2 = Q1_all.gather(1, action.unsqueeze(1)), Q2_all.gather(1, action.unsqueeze(1)) #[b,1]
        critic_loss = F.mse_loss(current_q1, target_Q) + F.mse_loss(current_q2, target_Q)
        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = current_q1.mean().item()
            metrics['critic_q2'] = current_q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return metrics
    
    def update_actor_and_alpha(self, obs, step):
        metrics = dict()

        probs = self.actor(obs) # [B, action_dim]
        log_probs = torch.log(probs + 1e-8)  # [B, action_dim]

        with torch.no_grad():
            actor_Q1, actor_Q2 = self.critic(obs) # [B, action_dim], [B, action_dim]

        min_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss =  torch.sum(probs * (self.alpha.detach()*log_probs - min_Q), dim=1, keepdim=False) # [B,]
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_ent'] = -log_probs.mean()
            metrics['target_entropy'] = self.target_entropy
            metrics['actor_log_probs'] = log_probs.mean().item()
            metrics['actor_loss'] = actor_loss.mean().item()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_probs - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            if self.use_tb or self.use_wandb:
                metrics['alpha_loss'] = alpha_loss.item()
                metrics['alpha_value'] = self.alpha.item()
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        metrics['batch_reward'] = reward.mean().item()

        metrics.update(self.update_critic(obs, action, reward, next_obs, discount, step))
        if step % self.actor_update_frequency == 0:
             # update critic
            metrics.update(self.update_actor_and_alpha(obs, step))

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)
            
        return metrics