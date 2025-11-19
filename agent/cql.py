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


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type
        self.action_dim = action_dim

        # Trunk for feature extraction
        if obs_type == 'pixels':
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), 
                nn.Tanh()
            )
        else:
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), 
                nn.Tanh()
            )

        # Q-networks that output Q-values for all actions
        self.q1_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        """Returns Q-values for all actions
        
        Args:
            obs: [batch_size, obs_dim]
        Returns:
            q1, q2: [batch_size, action_dim]
        """
        h = self.trunk(obs)
        q1 = self.q1_net(h)
        q2 = self.q2_net(h)
        return q1, q2


class CQLAgent:
    def __init__(self,
                 name,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 hidden_dim,
                 critic_target_tau,
                 epsilon_schedule,
                 num_expl_steps,
                 nstep,
                 batch_size,
                 use_tb,
                 alpha,
                 n_samples,
                 target_cql_penalty,
                 use_critic_lagrange,
                 has_next_action=False,
                 meta_dim=0):
        self.obs_type = obs_type
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.use_critic_lagrange = use_critic_lagrange
        self.target_cql_penalty = target_cql_penalty
        self.epsilon_schedule = epsilon_schedule
        self.num_expl_steps = num_expl_steps # Not used but kept for compatibility

        self.alpha = alpha
        self.n_samples = n_samples

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

        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                                    hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # lagrange multipliers
        self.log_critic_alpha = torch.zeros(1,
                                            requires_grad=True,
                                            device=device)

        # optimizers
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_alpha_opt = torch.optim.Adam([self.log_critic_alpha],
                                                 lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)
    
    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    
    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        
        # ε-greedy exploration
        epsilon = 0.0 if eval_mode else utils.schedule(self.epsilon_schedule, step)
        
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                q1, q2 = self.critic(inpt)
                q_values = torch.min(q1, q2)
                if eval_mode:
                    return q_values.argmax(dim=-1).item()
                else:
                    action = torch.multinomial(torch.softmax(q_values, dim=-1), num_samples=1).item()
        
        return action

    def aug_and_encode(self, obs):
        if self.obs_type == 'pixels':
            obs = self.aug(obs)
        return self.encoder(obs)

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Compute target Q-value
        with torch.no_grad():
            next_q1, next_q2 = self.critic_target(next_obs)
            next_q = torch.min(next_q1, next_q2)
            target_V = next_q.max(dim=1, keepdim=True)[0]
            target_Q = reward + (discount * target_V)

        # Get current Q-values
        current_q1, current_q2 = self.critic(obs)
        
        # Get Q-values for taken actions
        if action.dtype != torch.long:
            action = action.long()
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
            
        Q1 = current_q1.gather(1, action)
        Q2 = current_q2.gather(1, action)
        
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # ============ CQL PENALTY - 4 GRUPPI ============
        batch_size = obs.shape[0]
        
        # 1. RANDOM ACTIONS: campiona azioni casuali uniformemente
        # Shape: [n_samples, batch_size, 1]
        random_actions_idx = torch.randint(0, self.action_dim, 
                                          (self.n_samples, batch_size, 1),
                                          device=self.device)
        
        # 2. CURRENT POLICY ACTIONS su obs: campiona usando min(Q1, Q2) come in act()
        with torch.no_grad():
            # Usa il MINIMO tra Q1 e Q2 (non la media!)
            current_q_min = torch.min(current_q1, current_q2)  # [batch, action_dim]
            current_probs = F.softmax(current_q_min, dim=-1)  # [batch, action_dim]
            
            # Campiona n_samples azioni per ogni elemento del batch
            sampled_actions_idx = torch.multinomial(
                current_probs, 
                num_samples=self.n_samples, 
                replacement=True
            ).transpose(0, 1).unsqueeze(-1)  # [n_samples, batch, 1]
        
        # 3. NEXT POLICY ACTIONS: campiona da next_obs usando min(Q1, Q2)
        with torch.no_grad():
            # Valuta next_obs con il critic corrente (non target!)
            next_q1_full, next_q2_full = self.critic(next_obs)
            # Usa il MINIMO tra Q1 e Q2
            next_q_min = torch.min(next_q1_full, next_q2_full)  # [batch, action_dim]
            next_probs = F.softmax(next_q_min, dim=-1)
            
            next_sampled_actions_idx = torch.multinomial(
                next_probs,
                num_samples=self.n_samples,
                replacement=True
            ).transpose(0, 1).unsqueeze(-1)  # [n_samples, batch, 1]
        
        # ============ ESTRAZIONE Q-VALUES ============
        # IMPORTANTE: tutti i Q-values vengono valutati su obs (non next_obs)
        # Espandi current_q per broadcasting
        current_q1_expanded = current_q1.unsqueeze(0).expand(self.n_samples, -1, -1)
        current_q2_expanded = current_q2.unsqueeze(0).expand(self.n_samples, -1, -1)
        
        # Estrai Q-values per le 3 tipologie di azioni
        # Tutte valutate su obs corrente
        rand_Q1 = current_q1_expanded.gather(2, random_actions_idx)  # [n_samples, batch, 1]
        rand_Q2 = current_q2_expanded.gather(2, random_actions_idx)
        
        sampled_Q1 = current_q1_expanded.gather(2, sampled_actions_idx)  # [n_samples, batch, 1]
        sampled_Q2 = current_q2_expanded.gather(2, sampled_actions_idx)
        
        # next_sampled_actions vengono valutate su obs (come in CQL_OLD!)
        next_sampled_Q1 = current_q1_expanded.gather(2, next_sampled_actions_idx)  # [n_samples, batch, 1]
        next_sampled_Q2 = current_q2_expanded.gather(2, next_sampled_actions_idx)

        # ============ CONCATENAZIONE ============
        # Concatena tutti i 4 gruppi di Q-values lungo dim=0
        # Shape finale: [(3*n_samples + 1), batch, 1]
        cat_Q1 = torch.cat([
            rand_Q1,              # Random actions valutate su obs
            sampled_Q1,           # Current policy actions su obs
            next_sampled_Q1,      # Next policy actions valutate su obs
            Q1.unsqueeze(0)       # Dataset action
        ], dim=0)
        
        cat_Q2 = torch.cat([
            rand_Q2,
            sampled_Q2,
            next_sampled_Q2,
            Q2.unsqueeze(0)
        ], dim=0)

        # ============ CQL PENALTY ============
        # logsumexp su dim=0: somma su tutti i gruppi di azioni
        cql_logsumexp = torch.logsumexp(cat_Q1, dim=0).mean() + \
                        torch.logsumexp(cat_Q2, dim=0).mean()
        
        # Penalty = logsumexp(Q_all_actions) - Q_dataset
        cql_penalty = cql_logsumexp - (Q1 + Q2).mean()

        # Update lagrange multiplier
        if self.use_critic_lagrange:
            alpha = torch.clamp(self.log_critic_alpha.exp(),
                                min=0.0,
                                max=1000000.0)
            alpha_loss = -0.5 * alpha * (cql_penalty - self.target_cql_penalty)

            self.critic_alpha_opt.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.critic_alpha_opt.step()
            alpha = torch.clamp(self.log_critic_alpha.exp(),
                                min=0.0,
                                max=1000000.0).detach()
        else:
            alpha = self.alpha

        # Combine losses
        critic_loss = critic_loss + alpha * cql_penalty

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['critic_cql'] = cql_penalty.item()
            metrics['critic_cql_logsum'] = cql_logsumexp.item()

        return metrics

    def update_actor(self, obs, action, step):
        """No actor update for discrete action CQL - kept for compatibility"""
        metrics = dict()
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor (empty for compatibility)
        metrics.update(self.update_actor(obs, action, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics