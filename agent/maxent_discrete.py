import copy

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from torch.distributions.categorical import Categorical

import utils
from agent.ddpg_discrete import DDPGAgent


class MaxEntAgent(DDPGAgent):
    """Discrete DDPG with rollout-fitted MaxEnt density rewards."""

    def __init__(self,
                 maxent_scale=1.0,
                 maxent_eps=1e-3,
                 maxent_bandwidth=0.1,
                 maxent_kernel="epanechnikov",
                 maxent_pca_components=32,
                 maxent_rollout_steps=10000,
                 maxent_rollout_every_steps=80000,
                 maxent_rollout_eval_mode=False,
                 maxent_rollout_batch_size=256,
                 maxent_num_rollouts=20,
                 maxent_policy_weighting="uniform",
                 maxent_geometric_gamma=0.90,
                 maxent_max_policies=0,
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
        self.maxent_pca_components = int(maxent_pca_components) if maxent_pca_components else 0
        self.maxent_rollout_steps = int(maxent_rollout_steps)
        self.maxent_rollout_every_steps = int(maxent_rollout_every_steps)
        self.maxent_rollout_eval_mode = maxent_rollout_eval_mode
        self.maxent_rollout_batch_size = int(maxent_rollout_batch_size)
        self.maxent_num_rollouts = int(maxent_num_rollouts)
        self.maxent_policy_weighting = maxent_policy_weighting
        self.maxent_geometric_gamma = maxent_geometric_gamma
        self.maxent_max_policies = int(maxent_max_policies)
        self.maxent_log_reward = maxent_log_reward
        self.maxent_reward_clip = maxent_reward_clip
        self.non_episodic_intrinsic_returns = non_episodic_intrinsic_returns
        self.update_encoder = update_encoder

        self.rollout_env = None
        self._last_rollout_step = None
        self._density_update_count = 0
        self._last_density_data_size = 0
        self._last_policy_mixture_weights = np.ones(1, dtype=np.float64)

        self.maxent_policies = []
        self.maxent_kde = None
        self.maxent_pca = None

    def insert_env(self, env):
        self.rollout_env = env

    def _encode_density_obs(self, obs):
        obs = torch.as_tensor(obs, device=self.device).float()
        return self.encoder(obs)

    def _policy_weights(self):
        n_policies = len(self.maxent_policies) + 1  # component 0 is random.
        if n_policies == 1:
            return np.ones(1, dtype=np.float64)

        if self.maxent_policy_weighting == "uniform":
            weights = np.ones(n_policies, dtype=np.float64) / float(n_policies)
        elif self.maxent_policy_weighting == "geometric":
            weights = np.array([
                self.maxent_geometric_gamma ** (n_policies - i)
                for i in range(n_policies)
            ], dtype=np.float64)
            weights = np.abs(weights) / weights.sum()
        else:
            raise ValueError(f"Unsupported MaxEnt policy weighting: {self.maxent_policy_weighting}")

        return weights

    def _snapshot_current_policy(self):
        encoder = copy.deepcopy(self.encoder).to(self.device)
        actor = copy.deepcopy(self.actor).to(self.device)
        encoder.train(False)
        actor.train(False)
        for param in encoder.parameters():
            param.requires_grad = False
        for param in actor.parameters():
            param.requires_grad = False

        self.maxent_policies.append((encoder, actor))
        if self.maxent_max_policies > 0:
            self.maxent_policies = self.maxent_policies[-self.maxent_max_policies:]

    def _act_from_snapshot(self, policy, obs):
        encoder, actor = policy
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            rep = encoder(obs)
            probs = actor(rep)
            if self.maxent_rollout_eval_mode:
                action = probs.argmax(dim=-1).item()
            else:
                action = Categorical(probs).sample().item()
        return action

    def _select_mixture_action(self, obs):
        weights = self._policy_weights()
        idx = np.random.choice(np.arange(len(weights)), p=weights)
        self._last_policy_mixture_weights = weights
        if idx == 0:
            return np.random.randint(self.action_dim)
        return self._act_from_snapshot(self.maxent_policies[idx - 1], obs)

    def _fit_density_model(self, data):
        if data.size == 0:
            return

        self.maxent_pca = None
        fit_data = data
        if 0 < self.maxent_pca_components < data.shape[1]:
            n_components = min(self.maxent_pca_components, data.shape[0], data.shape[1])
            if n_components > 0:
                self.maxent_pca = PCA(n_components=n_components, whiten=False)
                fit_data = self.maxent_pca.fit_transform(data)

        self.maxent_kde = KernelDensity(
            bandwidth=self.maxent_bandwidth,
            kernel=self.maxent_kernel,
        ).fit(fit_data)
        self._last_density_data_size = int(data.shape[0])

    def _estimate_density(self, rep):
        if self.maxent_kde is None:
            return torch.zeros(rep.shape[0], 1, device=rep.device)

        data = rep.detach().cpu().numpy()
        if self.maxent_pca is not None:
            data = self.maxent_pca.transform(data)

        density = np.exp(self.maxent_kde.score_samples(data))
        return torch.as_tensor(density, device=rep.device, dtype=rep.dtype).reshape(-1, 1)

    def compute_intr_reward(self, rep, step):
        density = self._estimate_density(rep)
        reward = 1.0 / (density + self.maxent_eps)
        if self.maxent_log_reward:
            reward = torch.log(reward + 1.0)
        reward = self.maxent_scale * reward
        if self.maxent_reward_clip and self.maxent_reward_clip > 0.0:
            reward = torch.clamp(reward, max=self.maxent_reward_clip)
        return reward, density

    def _collect_mixed_policy_rollouts(self):
        if self.rollout_env is None:
            raise RuntimeError(
                "MaxEntAgent requires insert_env(env) before reward-free updates "
                "so it can collect rollout states for density estimation."
            )

        observations = []
        was_training = self.training
        self.train(False)
        try:
            for _ in range(self.maxent_num_rollouts):
                meta = self.init_meta()
                time_step = self.rollout_env.reset()
                for _ in range(self.maxent_rollout_steps):
                    action = self._select_mixture_action(time_step.observation)
                    time_step = self.rollout_env.step(action)
                    observations.append(time_step.observation)
                    meta = self.update_meta(meta, self._last_rollout_step or 0, time_step)
                    if time_step.last():
                        time_step = self.rollout_env.reset()
                        meta = self.init_meta()
        finally:
            self.train(was_training)

        return observations

    def _rollout_features(self, observations):
        if not observations:
            return np.empty((0, self.obs_dim), dtype=np.float32)

        observations = np.stack(observations)
        reps = []
        for start in range(0, len(observations), self.maxent_rollout_batch_size):
            obs = observations[start:start + self.maxent_rollout_batch_size]
            with torch.no_grad():
                rep = self._encode_density_obs(obs)
                reps.append(rep.detach().cpu().numpy())

        return np.concatenate(reps, axis=0)

    def _update_density_from_mixture(self, step):
        if self.maxent_kde is not None:
            self._snapshot_current_policy()

        observations = self._collect_mixed_policy_rollouts()
        data = self._rollout_features(observations)
        self._fit_density_model(data)

        self._density_update_count += 1
        self._last_rollout_step = step

    def _maybe_update_density(self, step):
        if self.maxent_rollout_every_steps <= 0:
            return
        if self._last_rollout_step is not None:
            if step - self._last_rollout_step < self.maxent_rollout_every_steps:
                return
        self._update_density_from_mixture(step)

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
                self._maybe_update_density(step)
                density_rep = self._encode_density_obs(raw_next_obs)
                intr_reward, density = self.compute_intr_reward(density_rep, step)

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
                metrics['maxent_density_data_size'] = self._last_density_data_size
                metrics['maxent_density_updates'] = self._density_update_count
                metrics['maxent_num_policies'] = len(self.maxent_policies)
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
