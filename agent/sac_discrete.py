import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from agent import Agent
from utils import ColorPrint
from collections import OrderedDict

import utils
from agent.utils import Encoder, ActorDiscrete as Actor, CriticDiscrete as Critic


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
                 update_actor_after_critic_steps,
                 eps_schedule,
                 nstep,
                 init_temperature, 
                 alpha_lr, 
                 actor_lr, 
                 actor_update_frequency, 
                 critic_lr,
                 critic_target_tau, 
                 critic_target_update_frequency,
                 init_critic,
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
        self.init_critic = init_critic
        self.eps_schedule = eps_schedule
        self.update_actor_after_critic_steps = update_actor_after_critic_steps

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

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

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
        obs = self.encoder(obs.unsqueeze(0))
        inputs = [obs]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        with torch.no_grad():
            probs = self.actor(inpt)

            if eval_mode:
                action = probs.argmax(dim=-1).cpu().numpy()[0]
            else:
                eps = utils.schedule(self.eps_schedule, step)
                if np.random.rand() < eps:
                    action = np.random.randint(self.action_dim)
                else:
                    action = Categorical(probs).sample().item()
                if step < self.num_expl_steps:
                    if step % 1000 == 0 and step > 0:
                        ColorPrint.yellow("DDPG Discrete: using random action for exploration")
                    # sample the discrete action uniformly during initial exploration
                    action = np.random.randint(self.action_dim)
        return action
    
    

    def update_critic(self, obs, action, reward, next_obs, discount, step):
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
    
    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)
    
    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        metrics.update(self.update_critic(obs, action, reward, next_obs, discount, step))
        if step >= self.update_actor_after_critic_steps:
            if step % self.actor_update_frequency == 0:
                # update actor and alpha
                metrics.update(self.update_actor_and_alpha(obs.detach(), step))

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)
            
        return metrics