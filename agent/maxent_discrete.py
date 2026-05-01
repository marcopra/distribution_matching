import math

import numpy as np
import torch

import utils
from agent.ddpg_discrete import DDPGAgent


class MaxEntAgent(DDPGAgent):
    """Discrete DDPG with MaxEnt state-density rewards for reward-free runs."""

    def __init__(self,
                 maxent_scale=1.0,
                 maxent_eps=1e-3,
                 maxent_bandwidth=0.1,
                 maxent_kernel="epanechnikov",
                 maxent_buffer_size=4096,
                 maxent_rep_dim=32,
                 maxent_rollout_steps=1000,
                 maxent_rollout_every_steps=1000,
                 maxent_rollout_eval_mode=False,
                 maxent_rollout_batch_size=256,
                 maxent_log_reward=False,
                 maxent_reward_clip=0.0,
                 non_episodic_intrinsic_returns=False,
                 update_encoder=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.maxent_scale = maxent_scale
        self.maxent_eps = maxent_eps
        self.maxent_bandwidth = maxent_bandwidth
        self.maxent_kernel = maxent_kernel
        self.maxent_buffer_size = int(maxent_buffer_size)
        self.maxent_rep_dim = int(maxent_rep_dim) if maxent_rep_dim else 0
        self.maxent_rollout_steps = int(maxent_rollout_steps)
        self.maxent_rollout_every_steps = int(maxent_rollout_every_steps)
        self.maxent_rollout_eval_mode = maxent_rollout_eval_mode
        self.maxent_rollout_batch_size = int(maxent_rollout_batch_size)
        self.maxent_log_reward = maxent_log_reward
        self.maxent_reward_clip = maxent_reward_clip
        self.non_episodic_intrinsic_returns = non_episodic_intrinsic_returns
        self.update_encoder = update_encoder
        self.rollout_env = None
        self._last_rollout_step = None

        self.maxent_density_dim = self.obs_dim
        self.maxent_projection = None
        if 0 < self.maxent_rep_dim < self.obs_dim:
            self.maxent_density_dim = self.maxent_rep_dim
            self.maxent_projection = torch.randn(self.obs_dim,
                                                 self.maxent_density_dim,
                                                 device=self.device)
            self.maxent_projection /= math.sqrt(self.maxent_density_dim)

        self.maxent_buffer = torch.zeros(self.maxent_buffer_size,
                                         self.maxent_density_dim,
                                         device=self.device)
        self.maxent_buffer_ptr = 0
        self.maxent_buffer_full = False

    def insert_env(self, env):
        self.rollout_env = env

    def _density_features(self, rep):
        if self.maxent_projection is None:
            return rep
        return rep @ self.maxent_projection

    def _encode_density_obs(self, obs):
        obs = torch.as_tensor(obs, device=self.device).float()
        return self.encoder(obs)

    def _density_support(self):
        if self.maxent_buffer_full:
            return self.maxent_buffer
        if self.maxent_buffer_ptr == 0:
            return None
        return self.maxent_buffer[:self.maxent_buffer_ptr]

    def _enqueue_density_features(self, rep):
        rep = rep.detach()
        batch_size = rep.shape[0]
        if batch_size >= self.maxent_buffer_size:
            self.maxent_buffer.copy_(rep[-self.maxent_buffer_size:])
            self.maxent_buffer_ptr = 0
            self.maxent_buffer_full = True
            return

        ptr = self.maxent_buffer_ptr
        remaining = self.maxent_buffer_size - ptr
        if batch_size <= remaining:
            self.maxent_buffer[ptr:ptr + batch_size].copy_(rep)
        else:
            self.maxent_buffer[ptr:].copy_(rep[:remaining])
            self.maxent_buffer[:batch_size - remaining].copy_(rep[remaining:])

        self.maxent_buffer_ptr = (ptr + batch_size) % self.maxent_buffer_size
        if self.maxent_buffer_ptr <= ptr:
            self.maxent_buffer_full = True

    def _estimate_density(self, rep):
        support = self._density_support()
        if support is None:
            return torch.zeros(rep.shape[0], 1, device=rep.device)

        bandwidth = max(float(self.maxent_bandwidth), 1e-8)
        dist = torch.cdist(rep, support, p=2) / bandwidth

        if self.maxent_kernel == "gaussian":
            kernel_values = torch.exp(-0.5 * dist.square())
        elif self.maxent_kernel == "epanechnikov":
            kernel_values = torch.clamp(1.0 - dist.square(), min=0.0)
        else:
            raise ValueError(f"Unsupported MaxEnt kernel: {self.maxent_kernel}")

        return kernel_values.mean(dim=1, keepdim=True)

    def compute_intr_reward(self, rep, step):
        density = self._estimate_density(rep)
        reward = 1.0 / (density + self.maxent_eps)
        if self.maxent_log_reward:
            reward = torch.log(reward + 1.0)
        reward = self.maxent_scale * reward
        if self.maxent_reward_clip and self.maxent_reward_clip > 0.0:
            reward = torch.clamp(reward, max=self.maxent_reward_clip)
        return reward, density

    def _collect_rollout_density(self, step):
        if self.rollout_env is None:
            raise RuntimeError(
                "MaxEntAgent requires insert_env(env) before reward-free updates "
                "so it can collect rollout states for density estimation."
            )

        observations = []
        meta = self.init_meta()
        time_step = self.rollout_env.reset()

        was_training = self.training
        self.train(False)
        try:
            for _ in range(self.maxent_rollout_steps):
                with torch.no_grad():
                    action = self.act(time_step.observation,
                                      meta,
                                      step,
                                      eval_mode=self.maxent_rollout_eval_mode)
                time_step = self.rollout_env.step(action)
                observations.append(time_step.observation)
                meta = self.update_meta(meta, step, time_step)
                if time_step.last():
                    time_step = self.rollout_env.reset()
                    meta = self.init_meta()
        finally:
            self.train(was_training)

        if not observations:
            return

        observations = np.stack(observations)
        reps = []
        for start in range(0, len(observations), self.maxent_rollout_batch_size):
            obs = observations[start:start + self.maxent_rollout_batch_size]
            with torch.no_grad():
                rep = self._encode_density_obs(obs)
                reps.append(self._density_features(rep))

        self._enqueue_density_features(torch.cat(reps, dim=0))

    def _maybe_collect_rollout_density(self, step):
        if self.maxent_rollout_every_steps <= 0:
            return
        if self._last_rollout_step is not None:
            if step - self._last_rollout_step < self.maxent_rollout_every_steps:
                return

        self._collect_rollout_density(step)
        self._last_rollout_step = step

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        raw_next_obs = next_obs

        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            with torch.no_grad():
                self._maybe_collect_rollout_density(step)
                density_rep = self._encode_density_obs(raw_next_obs)
                density_features = self._density_features(density_rep)
                intr_reward, density = self.compute_intr_reward(density_features, step)

            reward = intr_reward
            critic_discount = (torch.ones_like(discount) if self.non_episodic_intrinsic_returns
                               else discount)
        else:
            reward = extr_reward
            critic_discount = discount
            density = None

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()
            if self.reward_free:
                metrics['intr_reward'] = intr_reward.mean().item()
                metrics['maxent_density'] = density.mean().item()
                metrics['maxent_buffer_size'] = (self.maxent_buffer_size if self.maxent_buffer_full
                                                 else self.maxent_buffer_ptr)
                metrics['maxent_last_rollout_step'] = (-1 if self._last_rollout_step is None
                                                       else self._last_rollout_step)

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        metrics.update(
            self.update_critic(obs.detach(), action, reward, critic_discount,
                               next_obs.detach(), step))

        if step >= self.update_actor_after_critic_steps:
            metrics.update(self.update_actor(obs.detach(), step))

        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
