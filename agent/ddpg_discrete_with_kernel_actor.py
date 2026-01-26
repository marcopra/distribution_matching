import hydra
import utils
import torch
import numpy as np
import torch.nn as nn
from utils import ColorPrint
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions.categorical import Categorical
from agent.utils import KernelActorDiscrete as KernelActor, CriticDiscrete as Critic
# from agent.dist_matching_embedding import Encoder as KernelEncoder # TODO decidere il dtype da usare perchè in dist_matching uso float64 ma in tutto il resto float32

class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, feature_dim):
        super(Encoder, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.repr_dim = feature_dim
        self.temperature = 0.05

        # self.fc = nn.Identity()
        # self.fc = nn.Linear(obs_shape[0], feature_dim, bias=False)
        self.fc =  nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=False),
            # nn.LayerNorm(feature_dim),
        )

        # nn.init.eye_(self.fc[0].weight)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], -1)
        h = self.fc(obs)
        h = F.normalize(h, p=1, dim=-1)
        # h = F.normalize(h, p=2, dim=-1)
        # h = F.softmax(h/self.temperature, dim=-1)

        return h
    
    def encode_and_project(self, obs):
        h = self.forward(obs)
        return h


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3

        self.conv = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU())

        self.repr_dim = 32 * 35 * 35

        self.projector = nn.Sequential(
            nn.Linear(self.repr_dim, feature_dim),
            # nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.
        h = self.conv(obs)
        h = h.view(h.shape[0], -1)
        # h = F.softmax(h/0.1, dim=-1)
        return h

    def encode_and_project(self, obs):
        h = self.forward(obs)
        z = self.projector(h)
        z =F.normalize(z, p=1, dim=-1)
        return z
    
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
                 dataset_dim,
                 eta,
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
        self.eps_schedule = eps_schedule
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.update_actor_after_critic_steps = num_expl_steps + update_actor_after_critic_steps
        self.dataset_dim = dataset_dim
        self.eta = eta

        # raise NotImplementedError("DDPGAgent with Kernel Actor is not fully implemented yet.")
        # Bisogna modificare l'inizializzazione del kernel actor (al momento lavorare con identità)
        # e modificare inizializzazione del kernel actor
        # models
        if obs_type == 'pixels':
            self.aug = nn.Identity()  # TODO: implement data augmentation for pixels
            self.encoder = CNNEncoder(obs_shape, feature_dim).to(self.device)
            self.obs_dim = feature_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = Encoder(
                obs_shape, 
                hidden_dim, 
                self.feature_dim
            ).to(self.device) # KernelEncoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
            # self.obs_dim = self.obs_shape[0] + meta_dim

        self.actor = KernelActor(obs_type, self.obs_dim, self.dataset_dim, self.action_dim, self.eta).to(device)

        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = None #torch.optim.Adam(self.encoder.parameters(), lr=lr)
       
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
        # Caso 1: Altro DDPGAgent (comportamento esistente)
        if type(other).__name__ == 'DDPGAgent':
            raise NotImplementedError("DDPGAgent init_from another DDPGAgent is not implemented yet.")
            utils.hard_update_params(other.encoder, self.encoder)
            utils.hard_update_params(other.actor, self.actor)
            if self.init_critic:
                utils.hard_update_params(other.critic.trunk, self.critic.trunk)
            print("✓ Initialized from another DDPG agent")
        
        # Caso 2: DistMatchingEmbeddingAgent
        elif type(other).__name__ == 'DistMatchingEmbeddingAgent':
            # Carica encoder
            self.encoder.load_state_dict(other.encoder.state_dict())
            print("✓ Encoder loaded from DistMatchingEmbeddingAgent")
            
            # Inizializza kernel actor se disponibile
            if not hasattr(other, '_phi_all_obs'):
                raise RuntimeError(
                    "DistMatchingEmbeddingAgent not fully trained. "
                    "Missing cached features (_phi_all_obs). "
                    "Make sure the agent completed at least one policy update."
                )
            
            # Extract E matrix (action one-hot encoding)
            # E shape: [num_unique, n_actions]
            E = other.E  # This should be available from the cached features
            
            self.actor.initialize_from_pretrained(
                phi_dataset=other._phi_all_obs.to(self.device),
                gradient_coeff=other.gradient_coeff.to(self.device),
                eta=other.lr_actor,
                E=E.to(self.device)  # Pass E matrix for proper initialization
            )
            print("✓ KernelActorDiscrete initialized from pretrained weights")
            print(f"  Dataset size: {other.dataset.size}")
            print(f"  Feature dim: {other.feature_dim}")
            print(f"  Eta: {other.lr_actor}")
            print(f"  E matrix shape: {E.shape}")
                  
        else:
            raise ValueError(
                f"Cannot init_from agent of type {type(other).__name__}. "
                f"Expected DDPGAgent or DistMatchingEmbeddingAgent."
            )
        # copy parameters over
        # utils.hard_update_params(other.encoder, self.encoder)
        # utils.hard_update_params(other.actor, self.actor)
        # if self.init_critic:
        #     utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode):
        obs = torch.FloatTensor(obs).to(self.device)

        obs = self.aug_and_encode(obs.unsqueeze(0), project=True)
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
                if self.eps_schedule is not None:
                    eps = utils.schedule(self.eps_schedule, step)
                    if np.random.rand() < eps:
                        action = np.random.randint(self.action_dim)
                    else:
                        action = Categorical(probs).sample().item()
                else:
                    action = Categorical(probs).sample().item()
                    if step < self.num_expl_steps and self.reward_free:
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
            
            target_Q = target_Q.to(target_Q1.dtype)  # TODO for the future ensure all models use the same dtype

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

    def aug_and_encode(self, obs, project=False):
        obs = self.aug(obs)
        if project:
            return self.encoder.encode_and_project(obs)
        else:
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
        obs = self.aug_and_encode(obs, project=True)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs, project=True)

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
