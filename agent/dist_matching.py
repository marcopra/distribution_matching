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
    

class DistMatchingAgent:
    def __init__(self,
                 env
                 ):


        self.env = env
        self.n_states = env.unwrapped.n_states
        self.n_actions = env.action_space.n
        self.visited_states = set()
        # Get transition matrix from environment
        self.R_vector = np.zeros((self.n_states * self.n_actions, 1))
        self.R_tilde = np.zeros((self.n_states, 1))

        self.policy_operator = self._create_uniform_policy() 
        self.policy_matrix = self._from_operator_to_policy(self.policy_operator)
        self.training = False
    
    def train(self, training=True):
        self.training = training
    

    def init_meta(self):
        return OrderedDict()

    def get_meta_specs(self):
        return tuple()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    
    def _create_uniform_policy(self) -> np.ndarray:
        """Create uniform random policy operator."""
        P = np.zeros((self.n_states * self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row = s * self.n_actions + a
                P[row, s] = 1.0 / self.n_actions
                
        return P
    
    def _create_random_policy(self) -> np.ndarray:
        """Create random random policy operator."""
        P = np.zeros((self.n_states * self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row = s * self.n_actions + a
                P[row, s] = np.random.rand()
            P[:, s] /= P[:, s].sum()
        return P

    
    def _from_operator_to_policy(self, policy_operator: np.ndarray = None) -> np.ndarray:
        """Convert policy operator to policy matrix."""
        if policy_operator is None:
            policy_operator = self.policy_operator
        policy_matrix = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row = s * self.n_actions + a
                policy_matrix[s, a] = policy_operator[row, s]
        return policy_matrix
    
    def load_policy_operator(self, path: str):
        """Load policy operator from file."""
        self.policy_operator = np.load(path)
        self.policy_matrix = self._from_operator_to_policy(self.policy_operator)
        print(f"Loaded policy operator from: {path}")

    # def train(self, training=True):
    #     self.training = training
    #     self.encoder.train(training)
    #     self.critic.train(training)

    # def init_from(self, other):
    #     # copy parameters over
    #     utils.hard_update_params(other.encoder, self.encoder)
    #     if self.init_critic:
    #         utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    # def get_meta_specs(self):
    #     return tuple()

    # def init_meta(self):
    #     return OrderedDict()

    # def update_meta(self, meta, global_step, time_step, finetune=False):
    #     return meta

    def act(self, obs, meta, step, eval_mode):
        # One hot to state
        state = np.argmax(obs)
        action_probs = self.policy_matrix[state]
        action_probs = action_probs / np.sum(action_probs)  # Normalize probabilities, it could be slightly off due to numerical issues
        return np.random.choice(self.n_actions, p=action_probs)
       

    # def update_critic(self, obs, action, reward, discount, next_obs, step): # Named critic for compatibility with url algorithms
    #     """Double DQN update"""
    #     metrics = dict()

    #     with torch.no_grad():
    #         # DOUBLE DQN: usa Q-network per selezionare azione
    #         next_q1, next_q2 = self.critic(next_obs)
    #         next_q = torch.min(next_q1, next_q2)  # Use min for robustness
            
    #         # Seleziona best action con Q-network (online)
    #         best_actions = next_q.argmax(dim=1, keepdim=True)  # [batch, 1]
            
    #         # Valuta con Q-target (questo è il Double DQN!)
    #         target_q1, target_q2 = self.critic_target(next_obs)
    #         target_q = torch.min(target_q1, target_q2)
            
    #         # Prendi Q-value della best action
    #         next_q_value = target_q.gather(1, best_actions)  # [batch, 1]
            
    #         # Calcola target
    #         target_Q = reward + discount * next_q_value

    #     # Q correnti
    #     current_q1, current_q2 = self.critic(obs)
        
    #     # Prendi Q-values delle azioni effettivamente prese
    #     # action deve essere [batch, 1] con dtype long
    #     if action.dtype != torch.long:
    #         action = action.long()
    #     if len(action.shape) == 1:
    #         action = action.unsqueeze(1)
            
    #     current_q1_values = current_q1.gather(1, action)
    #     current_q2_values = current_q2.gather(1, action)
        
    #     # Loss MSE su entrambe le Q-networks
    #     q_loss = F.mse_loss(current_q1_values, target_Q) + \
    #              F.mse_loss(current_q2_values, target_Q)

    #     if self.use_tb or self.use_wandb:
    #         metrics['q_target'] = target_Q.mean().item()
    #         metrics['q1'] = current_q1_values.mean().item()
    #         metrics['q2'] = current_q2_values.mean().item()
    #         metrics['q_loss'] = q_loss.item()

    #     # Ottimizza
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.zero_grad(set_to_none=True)
    #     self.critic_opt.zero_grad(set_to_none=True)
    #     q_loss.backward()
    #     self.critic_opt.step()
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.step()
            
    #     return metrics

    # def update_actor(self, obs, step):
    #     """Nessun aggiornamento dell'actor in DQN"""
    #     metrics = dict()

    #     return metrics

    # def aug_and_encode(self, obs):
    #     if self.obs_type == 'pixels':
    #         obs = self.aug(obs)
    #     return self.encoder(obs)

    # def update(self, replay_iter, step):
    #     metrics = dict()
    #     #import ipdb; ipdb.set_trace()

    #     if step % self.update_every_steps != 0:
    #         return metrics

    #     batch = next(replay_iter)
    #     obs, action, reward, discount, next_obs = utils.to_torch(
    #         batch, self.device)

    #     # augment and encode
    #     obs = self.aug_and_encode(obs)
    #     with torch.no_grad():
    #         next_obs = self.aug_and_encode(next_obs)

    #     if self.use_tb or self.use_wandb:
    #         metrics['batch_reward'] = reward.mean().item()

    #     # update critic
    #     metrics.update(
    #         self.update_critic(obs, action, reward, discount, next_obs, step))

    #     # update actor
    #     metrics.update(self.update_actor(obs.detach(), step))

    #     # update critic target
    #     utils.soft_update_params(self.critic, self.critic_target,
    #                              self.critic_target_tau)

    #     return metrics
