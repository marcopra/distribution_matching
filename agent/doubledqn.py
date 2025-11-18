from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # For state-based observations, just pass through
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
            # For states, observations are already processed
            return obs

class QNetwork(nn.Module):
    """Q-Network per Double DQN - output: Q-values per tutte le azioni"""
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type
        self.action_dim = action_dim

        if obs_type == 'pixels':
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim),
                nn.LayerNorm(feature_dim), 
                nn.Tanh()
            )
            trunk_dim = feature_dim
        else:
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), 
                nn.Tanh()
            )
            trunk_dim = hidden_dim
        
  
        def make_q():
            q_layers = [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            # Output: Q-value per ogni azione
            q_layers += [nn.Linear(hidden_dim, action_dim)]
            return nn.Sequential(*q_layers)

        # Double Q-learning: due Q-networks
        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs):
        """Restituisce Q-values per tutte le azioni
        
        Args:
            obs: [batch_size, obs_dim]
        Returns:
            q1, q2: [batch_size, action_dim]
        """
        h = self.trunk(obs)
        q1 = self.Q1(h)
        q2 = self.Q2(h)
        return q1, q2
    

class DoubleDQNAgent:
    def __init__(self,
                 name,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 epsilon_schedule,
                 nstep,
                 batch_size,
                 init_critic,
                 use_tb,
                 use_wandb,
                 meta_dim=0,
                 mode='continuous'):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.epsilon_schedule = epsilon_schedule 
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.mode = mode

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


        self.critic = QNetwork(obs_type, self.obs_dim, self.action_dim,
                                   feature_dim, hidden_dim).to(device)
        self.critic_target = QNetwork(obs_type, self.obs_dim, self.action_dim,
                                 feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

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
            # Esplorazione random
            action = np.random.randint(0, self.action_dim)
        else:
            # Sfruttamento: scegli azione con Q-value massimo
            with torch.no_grad():
                q1, q2 = self.critic(inpt)
                # Usa il minimo tra Q1 e Q2 per robustezza
                q_values = torch.min(q1, q2)
                if eval_mode:
                    return q_values.argmax(dim=-1).item()
                else:
                    action = torch.multinomial(torch.softmax(q_values, dim=-1), num_samples=1).item()
        
        return action

    def update_critic(self, obs, action, reward, discount, next_obs, step): # Named critic for compatibility with url algorithms
        """Double DQN update"""
        metrics = dict()

        with torch.no_grad():
            # DOUBLE DQN: usa Q-network per selezionare azione
            next_q1, next_q2 = self.critic(next_obs)
            next_q = torch.min(next_q1, next_q2)  # Use min for robustness
            
            # Seleziona best action con Q-network (online)
            best_actions = next_q.argmax(dim=1, keepdim=True)  # [batch, 1]
            
            # Valuta con Q-target (questo è il Double DQN!)
            target_q1, target_q2 = self.critic_target(next_obs)
            target_q = torch.min(target_q1, target_q2)
            
            # Prendi Q-value della best action
            next_q_value = target_q.gather(1, best_actions)  # [batch, 1]
            
            # Calcola target
            target_Q = reward + discount * next_q_value

        # Q correnti
        current_q1, current_q2 = self.critic(obs)
        
        # Prendi Q-values delle azioni effettivamente prese
        # action deve essere [batch, 1] con dtype long
        if action.dtype != torch.long:
            action = action.long()
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
            
        current_q1_values = current_q1.gather(1, action)
        current_q2_values = current_q2.gather(1, action)
        
        # Loss MSE su entrambe le Q-networks
        q_loss = F.mse_loss(current_q1_values, target_Q) + \
                 F.mse_loss(current_q2_values, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['q_target'] = target_Q.mean().item()
            metrics['q1'] = current_q1_values.mean().item()
            metrics['q2'] = current_q2_values.mean().item()
            metrics['q_loss'] = q_loss.item()

        # Ottimizza
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
            
        return metrics

    def update_actor(self, obs, step):
        """Nessun aggiornamento dell'actor in DQN"""
        metrics = dict()

        return metrics

    def aug_and_encode(self, obs):
        if self.obs_type == 'pixels':
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

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
