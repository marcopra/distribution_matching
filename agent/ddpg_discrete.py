
import hydra
import utils
import torch
import numpy as np
import torch.nn as nn
from utils import ColorPrint
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions.categorical import Categorical
from agent.utils import Encoder, ActorDiscrete as Actor, CriticDiscrete as Critic



class DDPGAgent:
    def __init__(self,
                 name,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape, # Number of discrete actions
                 device,
                 actor_lr,
                 critic_lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 update_actor_after_critic_steps,
                 eps_schedule,
                 nstep,
                 batch_size,
                 init_critic,
                 use_tb,
                 use_wandb,
                 linear_actor=False,
                 meta_dim=0):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.update_actor_after_critic_steps = update_actor_after_critic_steps
        self.eps_schedule = eps_schedule
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None

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

        # optimizers            

        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=self.actor_lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
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
    

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            next_probs = self.actor(next_obs)
            next_log_probs = torch.log(next_probs + 1e-8)
            target_Q1, target_Q2 = self.critic_target(next_obs)    
            target_V = torch.sum(next_probs * (torch.min(target_Q1,target_Q2) ), dim=1, keepdim=True) # [B,1]
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

       

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        probs = self.actor(obs) # [B, action_dim]
        log_probs = torch.log(probs + 1e-8)  # [B, action_dim]
        with torch.no_grad():
            actor_Q1, actor_Q2 = self.critic(obs) # [B, action_dim], [B, action_dim]

        min_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss =  - torch.sum(probs * min_Q, dim=1, keepdim=False) # [B,]


        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.mean().backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.mean().item()
            metrics['actor_logprob'] = log_probs.mean().item()
            metrics['actor_ent'] = -log_probs.mean()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()
        #import ipdb; ipdb.set_trace()

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

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        if step >= self.update_actor_after_critic_steps:
            # update actor
            metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
