import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils import ColorPrint

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


class ActorDiscrete(nn.Module):
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

    def _logits(self, obs):
        # Helper to get logits without softmax
        h = self.trunk(obs)
        return self.policy(h)

    def get_log_p(self, states, actions):
        """
        states:  (T, obs_dim) or (batch, obs_dim)
        actions: (T,) or (batch,) float with action indices
        returns: (T,) log-probabilities log pi(a_t | s_t)
        """
        logits = self._logits(states)                      # (T, K)
        log_probs = F.log_softmax(logits, dim=-1)          # (T, K)
        # convert actions to int64 for gather
        actions = actions.long()             # (T, 1)
        # Gather the log-prob of the taken action at each step
        log_p = log_probs.gather(dim=1, index=actions)     # (T, 1)
        return log_p.squeeze(-1)                           # (T,)

class CriticDiscrete(nn.Module):
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
    


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

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

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
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
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


# ****From MEPOL ****

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Correzione qui



from collections import OrderedDict
# torch.set_default_tensor_type(torch.DoubleTensor)
float_type = torch.float64

int_type = torch.int64
eps = 1e-7

class GaussianPolicy(nn.Module):
    """
    Gaussian Policy with state-independent diagonal covariance matrix
    """

    def __init__(self, hidden_sizes, num_features, action_dim, log_std_init=-0.5, activation=nn.ReLU):
        super().__init__()

        self.activation = activation

        layers = []
        layers.extend((nn.Linear(num_features, hidden_sizes[0]), self.activation()))
        for i in range(len(hidden_sizes) - 1):
            layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()))

        self.net = nn.Sequential(*layers)

        self.mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(log_std_init * torch.ones(action_dim, dtype=float_type))

        # Constants
        self.log_of_two_pi = torch.tensor(np.log(2*np.pi), dtype=float_type)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.mean.weight)

        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)

    def get_log_p(self, states, actions):
        mean, _ = self(states)
        return torch.sum(
            -0.5 * (
                self.log_of_two_pi
                + 2*self.log_std
                + ((actions - mean)**2 / (torch.exp(self.log_std) + eps)**2)
            ), dim=1
        )

    def forward(self, x, deterministic=False):
        mean = self.mean(self.net(x))

        if deterministic:
            output = mean
        else:
            output = mean + torch.randn(mean.size(), dtype=float_type) * torch.exp(self.log_std)

        return mean, output


    def predict(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=float_type).unsqueeze(0)
            return self(s, deterministic=deterministic)[1][0]


def train_supervised(env, policy, train_steps=100, batch_size=5000):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.00025)
    dict_like_obs = True if type(env.observation_space.sample()) is OrderedDict else False

    for _ in range(train_steps):
        optimizer.zero_grad()

        if dict_like_obs:
            states = torch.tensor([env.observation_space.sample()['observation'] for _ in range(5000)], dtype=float_type)
        else:
            states = torch.tensor([env.observation_space.sample()[:env.num_features] for _ in range(5000)], dtype=float_type)

        actions = policy(states)[0]
        loss = torch.mean((actions - torch.zeros_like(actions, dtype=float_type)) ** 2)

        loss.backward()
        optimizer.step()

    return policy