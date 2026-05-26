import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import ColorPrint
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from dm_env import StepType, specs
from copy import deepcopy

import numpy as np
from time import time
import os
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per salvare senza display
import matplotlib.pyplot as plt
from PIL import Image

# torch.set_default_tensor_type(torch.FloatTensor)
float_type = torch.float32


@dataclass(frozen=True)
class RawActorUpdateData:
    """Raw observations/actions for one actor update."""
    full: tuple
    source: str = "unknown"
    subsample: Optional[tuple] = None


@dataclass(frozen=True)
class EncodedActorUpdateData:
    """Pre-encoded features for one actor update."""
    full: Dict[str, torch.Tensor]
    rewards: torch.Tensor
    source: str = "unknown"
    subsample: Optional[Dict[str, torch.Tensor]] = None
    subsample_rewards: Optional[torch.Tensor] = None


class PointMazeNystromDebugHelper:
    """Build fixed PointMaze landmark transitions for Nyström debugging."""

    def __init__(self, border_margin: float = 0.05, oversample: float = 2.0):
        self.border_margin = float(border_margin)
        self.oversample = float(oversample)
        self.wrapped_env = None
        self.env = None
        self._subsample_batch = None
        self._fixed_xy_points = None
        self._fixed_actions = None

    @property
    def fixed_xy_points(self):
        return self._fixed_xy_points

    @property
    def fixed_actions(self):
        return self._fixed_actions

    def __getstate__(self):
        state = self.__dict__.copy()
        state["wrapped_env"] = None
        state["env"] = None
        return state

    def attach_env(self, env):
        self.wrapped_env = env
        self.env = self._find_discrete_env(env)
        self.clear_cache()

    def clear_cache(self):
        self._subsample_batch = None
        self._fixed_xy_points = None
        self._fixed_actions = None

    @staticmethod
    def _find_discrete_env(env):
        current = env
        while current is not None:
            if all(hasattr(current, attr) for attr in ("n_states", "idx_to_state", "state_to_idx")):
                return current
            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "unwrapped") and current.unwrapped is not current:
                current = current.unwrapped
            else:
                break
        return getattr(env, "unwrapped", env)

    def _iter_env_chain(self):
        current = self.wrapped_env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            yield current
            current = getattr(current, "env", None)

    def _env_method(self, method_name):
        for current in self._iter_env_chain():
            method = getattr(current, method_name, None)
            if callable(method):
                return method
        return None

    def _base_and_point_env(self):
        if self.wrapped_env is None:
            raise RuntimeError("PointMaze Nyström grid requires insert_env(env) before actor updates.")
        base_env = getattr(self.wrapped_env, "unwrapped", None)
        point_env = getattr(base_env, "point_env", None)
        if base_env is None or point_env is None:
            raise RuntimeError("PointMaze Nyström grid requires unwrapped.point_env.")
        return base_env, point_env

    def _set_state(self, xy, velocity=(0.0, 0.0)):
        base_env, point_env = self._base_and_point_env()
        qpos = point_env.data.qpos.copy()
        qvel = np.zeros_like(point_env.data.qvel)
        qpos[:2] = np.asarray(xy, dtype=np.float64)
        qvel[:2] = np.asarray(velocity, dtype=np.float64)
        point_env.set_state(qpos, qvel)
        if hasattr(base_env, "update_target_site_pos"):
            base_env.update_target_site_pos()

    @contextmanager
    def _preserve_state(self):
        base_env, point_env = self._base_and_point_env()
        snapshot = {
            "qpos": point_env.data.qpos.copy(),
            "qvel": point_env.data.qvel.copy(),
            "wrappers": [],
        }
        for current in self._iter_env_chain():
            wrapper_state = {}
            if hasattr(current, "_frames"):
                wrapper_state["frames"] = [frame.copy() for frame in list(current._frames)]
            if hasattr(current, "_cached_hidden_render"):
                cached = current._cached_hidden_render
                wrapper_state["cached_hidden_render"] = None if cached is None else cached.copy()
            if wrapper_state:
                snapshot["wrappers"].append((current, wrapper_state))

        try:
            yield
        finally:
            point_env.set_state(snapshot["qpos"], snapshot["qvel"])
            if hasattr(base_env, "update_target_site_pos"):
                base_env.update_target_site_pos()
            for wrapper, wrapper_state in snapshot["wrappers"]:
                if "frames" in wrapper_state and hasattr(wrapper, "_frames"):
                    wrapper._frames.clear()
                    wrapper._frames.extend([frame.copy() for frame in wrapper_state["frames"]])
                if "cached_hidden_render" in wrapper_state and hasattr(wrapper, "_cached_hidden_render"):
                    cached = wrapper_state["cached_hidden_render"]
                    wrapper._cached_hidden_render = None if cached is None else cached.copy()

    def _proprio_observation(self) -> np.ndarray:
        base_env, _ = self._base_and_point_env()
        point_obs, _ = base_env.point_env._get_obs()
        raw_obs = base_env._get_obs(point_obs)
        process_fn = self._env_method("_process_proprio_obs")
        if callable(process_fn):
            return np.asarray(process_fn(raw_obs), dtype=np.float32)

        if isinstance(raw_obs, dict):
            arrays = [
                np.asarray(value, dtype=np.float32).reshape(-1)
                for value in raw_obs.values()
                if not isinstance(value, str)
            ]
            return np.concatenate(arrays, dtype=np.float32)
        return np.asarray(raw_obs, dtype=np.float32)

    def _prepare_rendered_image(self, agent, image: np.ndarray, render_resolution: int) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        if agent.grayscale:
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[..., 0]
            elif image.ndim == 3:
                image = np.asarray(Image.fromarray(image).convert("L"))
            elif image.ndim != 2:
                raise ValueError(f"Expected grayscale image to be 2D or HWC, got shape {image.shape}")
        elif image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.shape[:2] != (render_resolution, render_resolution):
            image = np.asarray(
                Image.fromarray(image).resize((render_resolution, render_resolution), Image.LANCZOS)
            )

        if agent.grayscale and image.ndim == 2:
            image = image[..., None]
        elif not agent.grayscale and image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.ndim != 3 or image.shape[2] != agent.image_channels:
            raise ValueError(f"Expected image shape [H, W, {agent.image_channels}], got {image.shape}")
        return image

    def _observation(self, agent) -> np.ndarray:
        if agent.obs_type != "pixels":
            return self._proprio_observation().reshape(agent.obs_shape)

        render_fn = self._env_method("render_observation") or self._env_method("render")
        if not callable(render_fn):
            raise RuntimeError("PointMaze pixel Nyström grid requires render_observation() or render().")

        render_resolution = getattr(self.wrapped_env, "render_resolution", agent.obs_shape[-1])
        frame_stack = agent.obs_shape[0] // agent.image_channels
        image = self._prepare_rendered_image(agent, render_fn(), render_resolution)
        image_chw = image.transpose(2, 0, 1).copy()
        return np.tile(image_chw, (frame_stack, 1, 1))

    def maze_layout(self):
        layout_fn = self._env_method("get_debug_maze_layout")
        layout = layout_fn() if callable(layout_fn) else None
        layout = layout if isinstance(layout, dict) else self._layout_from_unwrapped_maze()
        if not isinstance(layout, dict):
            raise RuntimeError(
                "PointMaze Nyström grid requires get_debug_maze_layout() or maze.maze_map."
            )

        maze_lower = np.asarray(layout.get("maze_lower"), dtype=np.float32).reshape(-1)
        maze_upper = np.asarray(layout.get("maze_upper"), dtype=np.float32).reshape(-1)
        wall_rectangles = np.asarray(layout.get("wall_rectangles"), dtype=np.float32).reshape(-1, 4)
        if maze_lower.size != 2 or maze_upper.size != 2:
            raise RuntimeError("PointMaze layout must provide 2D maze_lower and maze_upper bounds.")
        return {
            "maze_lower": maze_lower[:2],
            "maze_upper": maze_upper[:2],
            "wall_rectangles": wall_rectangles,
        }

    def _layout_from_unwrapped_maze(self):
        base_env = getattr(self.wrapped_env, "unwrapped", None)
        maze = getattr(base_env, "maze", None)
        if maze is None or not hasattr(maze, "maze_map") or not hasattr(maze, "cell_rowcol_to_xy"):
            return None

        half_cell = 0.5 * float(getattr(maze, "maze_size_scaling", 1.0))
        all_rectangles, wall_rectangles = [], []
        for row_idx, row in enumerate(maze.maze_map):
            for col_idx, cell in enumerate(row):
                center = maze.cell_rowcol_to_xy(np.array([row_idx, col_idx], dtype=np.int32))
                rect = np.array(
                    [center[0] - half_cell, center[1] - half_cell, 2.0 * half_cell, 2.0 * half_cell],
                    dtype=np.float32,
                )
                all_rectangles.append(rect)
                if cell == 1:
                    wall_rectangles.append(rect)

        if not all_rectangles:
            return None
        all_rectangles = np.asarray(all_rectangles, dtype=np.float32)
        return {
            "maze_lower": all_rectangles[:, :2].min(axis=0),
            "maze_upper": (all_rectangles[:, :2] + all_rectangles[:, 2:4]).max(axis=0),
            "wall_rectangles": np.asarray(wall_rectangles, dtype=np.float32).reshape(-1, 4),
        }

    @staticmethod
    def _points_outside_walls(points: np.ndarray, wall_rectangles: np.ndarray, margin: float) -> np.ndarray:
        if wall_rectangles.size == 0:
            return np.ones(points.shape[0], dtype=bool)
        wall_lower = wall_rectangles[:, :2] - margin
        wall_upper = wall_rectangles[:, :2] + wall_rectangles[:, 2:4] + margin
        in_wall = ((points[:, None, :] >= wall_lower) & (points[:, None, :] <= wall_upper)).all(axis=2).any(axis=1)
        return ~in_wall

    @staticmethod
    def _xy_grid(lower: np.ndarray, upper: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
        xs = np.linspace(lower[0], upper[0], n_x, dtype=np.float32)
        ys = np.linspace(lower[1], upper[1], n_y, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def _fixed_start_xy(self):
        debug_fn = self._env_method("get_debug_coordinates")
        debug_info = debug_fn() if callable(debug_fn) else {}
        start = debug_info.get("fixed_start") if isinstance(debug_info, dict) else None
        return None if start is None else np.asarray(start, dtype=np.float32).reshape(-1)[:2]

    def build_grid_points(self, n_points: int) -> np.ndarray:
        if n_points <= 0:
            raise ValueError("Nyström PointMaze grid requires a positive number of points.")

        layout = self.maze_layout()
        margin = max(self.border_margin, 0.0)
        lower, upper = layout["maze_lower"] + margin, layout["maze_upper"] - margin
        if np.any(upper <= lower):
            lower, upper, margin = layout["maze_lower"], layout["maze_upper"], 0.0

        span = np.maximum(upper - lower, 1e-6)
        n_x = max(2, int(np.ceil(np.sqrt(n_points * (span[0] / span[1]) * self.oversample))))
        n_y = max(2, int(np.ceil(n_points * self.oversample / n_x)))

        valid_points = np.empty((0, 2), dtype=np.float32)
        for _ in range(8):
            candidates = self._xy_grid(lower, upper, n_x, n_y)
            valid_points = candidates[self._points_outside_walls(candidates, layout["wall_rectangles"], margin)]
            if valid_points.shape[0] >= n_points:
                break
            n_x, n_y = int(np.ceil(n_x * 1.4)) + 1, int(np.ceil(n_y * 1.4)) + 1

        if valid_points.shape[0] < n_points:
            raise RuntimeError(
                f"Could only place {valid_points.shape[0]} reachable PointMaze grid points; "
                f"requested {n_points}. Try reducing nystrom_grid_border_margin."
            )

        indices = np.linspace(0, valid_points.shape[0] - 1, n_points)
        selected = valid_points[np.round(indices).astype(np.int64)]

        start_xy = self._fixed_start_xy()
        if start_xy is not None and self._points_outside_walls(start_xy[None, :], layout["wall_rectangles"], margin)[0]:
            nearest = int(np.argmin(np.sum((selected - start_xy) ** 2, axis=1)))
            selected[nearest] = selected[0]
            selected[0] = start_xy
        return selected.astype(np.float32, copy=False)

    def _landmark_transition(self, agent, xy, action_idx):
        self._set_state(xy)
        obs = self._observation(agent)
        time_step = self.wrapped_env.step(int(action_idx))
        next_obs = self._observation(agent)
        reward = [float(getattr(time_step, "reward", 0.0))]
        discount = [float(getattr(time_step, "discount", 1.0))]
        return obs, next_obs, reward, discount

    def build_subsample_batch(self, agent):
        if self._subsample_batch is not None:
            return self._subsample_batch

        n_transitions = int(agent.subsamples if agent.subsamples is not None else agent.batch_size_actor)
        if n_transitions % agent.n_actions != 0:
            raise ValueError(
                f"PointMaze fixed debug dataset size={n_transitions} must be divisible by "
                f"n_actions={agent.n_actions} so each sampled state can include all actions. "
                "Set agent.subsamples or agent.batch_size_actor accordingly."
            )

        n_states = n_transitions // agent.n_actions
        state_points = self.build_grid_points(n_states)
        xy_points = np.repeat(state_points, agent.n_actions, axis=0)
        actions_np = np.tile(np.arange(agent.n_actions, dtype=np.int64), n_states)

        with self._preserve_state():
            transitions = [
                self._landmark_transition(agent, xy, action_idx)
                for xy, action_idx in zip(xy_points, actions_np)
            ]
        obs_list, next_obs_list, rewards, discounts = zip(*transitions)

        obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=agent.device)
        next_obs = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=agent.device)
        action = torch.as_tensor(actions_np, dtype=torch.long, device=agent.device)
        reward = torch.as_tensor(rewards, dtype=agent.compute_dtype, device=agent.device)
        discount = torch.as_tensor(discounts, dtype=agent.compute_dtype, device=agent.device)

        self._subsample_batch = (obs, action, reward, discount, next_obs)
        self._fixed_xy_points = state_points
        self._fixed_actions = actions_np
        dataset_name = "Nyström grid" if agent.subsamples is not None else "debug grid"
        ColorPrint.yellow(
            f"Using fixed PointMaze {dataset_name} with {n_states} reachable XY states "
            f"x {agent.n_actions} actions = {n_transitions} state-action landmarks "
            f"({agent._kernel_status()})."
        )
        self.save_fixed_points_plot(agent.n_actions)
        return self._subsample_batch

    def fixed_actor_batch(self, agent):
        obs, action, reward, _, next_obs = self.build_subsample_batch(agent)
        return agent._make_actor_batch(obs, action, next_obs, reward)

    def fixed_encoder_batch(self, agent):
        actor_batch = self.fixed_actor_batch(agent)
        size = min(int(agent.batch_size), actor_batch[0].shape[0])
        index = torch.randperm(actor_batch[0].shape[0], device=agent.device)[:size]
        return agent._slice_actor_batch(actor_batch, index)

    def encode_subsamples(self, agent):
        agent._sync_policy_encoder()
        encoded = agent._encode_actor_transition_batch_with_retries(self.build_subsample_batch(agent))
        return encoded, encoded.get("reward")

    def save_fixed_points_plot(self, n_actions: int):
        if self._fixed_xy_points is None:
            return
        from agent.utils_debug_visualization import PointMazeNystromDebugVisualizer

        PointMazeNystromDebugVisualizer().save_fixed_points_plot(
            layout=self.maze_layout(),
            points=self._fixed_xy_points,
            n_actions=n_actions,
        )

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        # compute representation dimension after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            dummy_output = self.convnet(dummy_input)
            self.repr_dim = dummy_output.view(1, -1).shape[1]

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

class KernelActorDiscrete(nn.Module):
    """
    Kernel-based actor that computes: π(a|s) = softmax(-η · (H^T · C_actions ⊙ E + C_bias))
    where:
    - H = [φ(s); 0] @ Φ_dataset^T  (augmented kernel similarities)
    - C_actions: gradient coefficients for state-action pairs [dataset_dim, n_actions]
    - C_bias: bias term for each action (last row of original gradient_coeff)
    - E: action one-hot encoding matrix [dataset_dim, n_actions]
    
    This architecture allows loading pretrained kernel weights and
    optionally finetuning them with RL algorithms.
    """
    
    def __init__(self, obs_type, input_dim, dataset_dim, action_dim, eta, trainable=True):
        """
        Args:
            obs_type: Type of observation ('states' or 'pixels')
            input_dim: Dimension of input features (d)
            dataset_dim: Number of dataset examples (n)
            action_dim: Number of actions
            eta: Scalar scaling factor (learning rate)
            trainable: If True, allows weights to be updated during finetuning
        """
        super().__init__()
        
        # Layer 1: Kernel layer computes H = [φ(x); 0] @ Φ_dataset^T
        # We need input_dim+1 to account for the augmented zero
        self.kernel_layer = nn.Linear(input_dim + 1, dataset_dim, bias=False, dtype=float_type)
        
        # Layer 2: Action-specific gradient coefficients
        # Shape: [dataset_dim, n_actions] corresponding to C[:-1] ⊙ E in original formulation
        self.action_coeffs = nn.Linear(dataset_dim, action_dim, bias=False, dtype=float_type)
        
        # Bias term: corresponds to C[-1] in original formulation
        # This is added uniformly to all actions
        self.bias_coeff = nn.Parameter(torch.zeros(action_dim, dtype=float_type))
        
        self.eta = nn.Parameter(torch.tensor(eta, dtype=float_type), requires_grad=trainable)
        self.softmax = nn.Softmax(dim=1)
        
        # Control whether weights are trainable
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
        
        self.apply(utils.weight_init)

    def initialize_from_pretrained(self, phi_dataset, gradient_coeff, eta, E=None):
        """
        Initialize weights from pretrained kernel policy.
        
        Args:
            phi_dataset: [num_unique, feature_dim+1] - augmented dataset feature matrix
            gradient_coeff: [num_unique+1, 1] - learned coefficients (last element is bias)
            eta: scalar - learning rate / temperature
            E: [num_unique, n_actions] - action one-hot encoding matrix (optional)
        """
        # 1. Initialize kernel layer: W = Φ_dataset (augmented with zeros)
        self.kernel_layer.weight.data.copy_(phi_dataset)
        
        # 2. Split gradient_coeff into action coeffs and bias
        # gradient_coeff shape: [num_unique+1, 1]
        # C[:-1] are action-specific coefficients, C[-1] is the bias
        action_grad = gradient_coeff[:-1].squeeze(-1)  # [num_unique]
        bias_grad = gradient_coeff[-1].item()  # scalar
        
        # 3. Initialize action_coeffs layer
        # We need to account for element-wise multiplication with E
        # Original: H @ (C[:-1] ⊙ E) where ⊙ is element-wise product
        # If E is provided, we can pre-compute C[:-1] ⊙ E
        
        # E shape: [num_unique, n_actions]
        # C[:-1] shape: [num_unique]
        # Broadcasting: C[:-1].unsqueeze(1) * E → [num_unique, n_actions]
        weighted_E = action_grad.unsqueeze(1) * E  # [num_unique, n_actions]
        # action_coeffs.weight shape: [n_actions, num_unique]
        # We want: logits = H @ weighted_E = H @ W^T, so W^T = weighted_E
        self.action_coeffs.weight.data.copy_(weighted_E.T)
    
        
        # 4. Initialize bias term
        self.bias_coeff.data.fill_(bias_grad)
        
        # 5. Set eta
        self.eta.data.copy_(torch.tensor(eta))
        print("all dtypes:", self.kernel_layer.weight.dtype, self.action_coeffs.weight.dtype, self.bias_coeff.dtype, self.eta.dtype)
        print(f"Kernel actor initialized from pretrained weights:")
        print(f"  - Kernel layer: {self.kernel_layer.weight.shape}")
        print(f"  - Action coeffs: {self.action_coeffs.weight.shape}")
        print(f"  - Bias: {self.bias_coeff.shape}")
        print(f"  - Eta: {self.eta.item()}")

    def forward(self, phi_x):
        """
        Forward pass matching dist_matching_embedding_augmented.py structure:
        
        1. Augment φ(x) with zero: [φ(x); 0]
        2. Compute kernel similarities: H = [φ(x); 0] @ Φ_dataset^T
        3. Apply gradient coefficients: 
           - action_logits = H @ (C[:-1] ⊙ E)  [via action_coeffs layer]
           - bias_logits = 1 * C[-1]             [via bias_coeff parameter]
        4. Combine: logits = action_logits + bias_logits
        5. Apply softmax: π(a|s) = softmax(-η * logits)
        
        Args:
            phi_x: [batch_size, feature_dim] - encoded observations
            
        Returns:
            probs: [batch_size, n_actions] - action probabilities
        """
        batch_size = phi_x.shape[0]
        
        # Step 1: Augment φ(x) con zero nell'ultima dimensione
        # Original: enc_obs_augmented = torch.cat([enc_obs, torch.zeros((1, 1))], dim=1)
        phi_x_aug = torch.cat([phi_x, torch.zeros(batch_size, 1, device=phi_x.device)], dim=1)
        phi_x_aug = phi_x_aug.to(dtype=float_type)
        # Shape: [batch_size, feature_dim + 1]
        
        # Step 2: Calcola le similarità del kernel H = [φ(x); 0] @ Φ_dataset^T
        # kernel_layer computes: H = phi_x_aug @ kernel_layer.weight^T
        h = self.kernel_layer(phi_x_aug)
        # Shape: [batch_size, dataset_dim]
        
        # Step 3a: Applica i coefficienti del gradiente specifici per azione
        # Original: H @ (self.gradient_coeff[:-1] * self.E)
        # action_coeffs.weight già contiene (C[:-1] ⊙ E)^T
        action_logits = self.action_coeffs(h)
        # Shape: [batch_size, n_actions]
        
        # Step 3b: Aggiungi il termine di bias (corrisponde a C[-1] nella formulazione originale)
        # Original: + torch.ones(1, self.E.shape[1]) * self.gradient_coeff[-1]
        bias_logits = self.bias_coeff.unsqueeze(0).expand(batch_size, -1)
        # Shape: [batch_size, n_actions]
        
        # Step 4: Combina i logit delle azioni e il bias
        logits = action_logits + bias_logits
        
        # Step 5: Scala per -eta e applica softmax
        # Original: torch.softmax(-self.lr_actor * (...), dim=1)
        probs = self.softmax(-self.eta * logits)
        
        return probs

    def _logits(self, phi_x):
        """
        Ottieni i logit senza softmax (utile per alcuni algoritmi RL).
        Segue lo stesso calcolo di forward() ma restituisce i logit grezzi.
        """
        batch_size = phi_x.shape[0]
        
        # Augmenta con zero
        phi_x_aug = torch.cat([phi_x, torch.zeros(batch_size, 1, device=phi_x.device)], dim=1)
        
        # Similarità del kernel
        h = self.kernel_layer(phi_x_aug)
        
        # Logit specifici per azione + bias
        action_logits = self.action_coeffs(h)
        bias_logits = self.bias_coeff.unsqueeze(0).expand(batch_size, -1)
        logits = action_logits + bias_logits
        
        # Scala per -eta (senza softmax)
        return -self.eta * logits

    def get_log_p(self, phi_x, actions):
        """
        Compute log probabilities for given actions.
        
        Args:
            phi_x: [T, feature_dim] - encoded states
            actions: [T] - action indices
            
        Returns:
            log_p: [T] - log probabilities
        """
        logits = self._logits(phi_x)  # [T, n_actions]
        log_probs = F.log_softmax(logits, dim=-1)  # [T, n_actions]
        
        # Gather log-prob of taken actions
        actions = actions.long().unsqueeze(1)  # [T, 1]
        log_p = log_probs.gather(dim=1, index=actions)  # [T, 1]
        
        return log_p.squeeze(-1)  # [T]
    
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


# ============================================================================
# Internal Dataset Management
# ============================================================================

class InternalDatasetFIFO:
    """
    FIFO-based internal dataset that maintains only the last N sampling periods.
    Each call to get_data() marks the end of a sampling period and retrieves
    data from the last N periods.
    """
    
    def __init__(self, dataset_type: str, n_states: int, n_actions: int, 
                 gamma: float, window_size: int, n_subsamples: int, 
                 subsampling_strategy: str, dynamic_horizon: bool = False,
                 obs_shape: tuple = None,
                 device: str = 'cpu', data_type=torch.double, first_state = None, second_state = None):
        """
        Args:
            dataset_type: "unique" or "all"
            n_states: Number of states
            n_actions: Number of actions
            gamma: Discount factor for geometric sampling
            window_size: Number of sampling periods to keep in memory
            n_subsamples: Number of samples to return per period (None = all)
            subsampling_strategy: Strategy for subsampling ("random" or "eder")
            dynamic_horizon: Whether to use a dynamic horizon
            obs_shape: Shape of observations (e.g., (84, 84, 3) for images)
            device: Torch device
        """
        self.dataset_type = dataset_type
        self.n_states = n_states
        self.n_actions = n_actions
        self.expected_size = n_states * n_actions
        assert dataset_type in ("unique", "all"), "dataset_type must be 'unique' or 'all'"
        self.n_subsamples = n_subsamples
        self.gamma = gamma
        self.window_size = window_size
        self.device = torch.device(device)
        self.dynamic_horizon = dynamic_horizon
        self.obs_shape = obs_shape if obs_shape is not None else (n_states,)
        self.data_type = data_type
        self.first_state = first_state
        self.second_state = second_state
        
        # FIFO queue: list of sampling periods, each period is a dict of tensors
        self._periods_queue = []
        self._current_period_data = None
        self._current_period_idx = 0
        self._last_period_size = 0  # Track size of last added period
        self.max_log_det = -np.inf
        self.subsampling_strategy = subsampling_strategy
        
        # Track horizons for dynamic horizon mode
        self._horizon_history = []
        self._plot_counter = 0
        
        # Cache for dummy transition (first complete sample)
        self._dummy_cache = None
        
        self.reset()
    
    def reset(self):
        """Reset the FIFO dataset."""
        utils.ColorPrint.yellow("Resetting FIFO internal dataset.")
        self._periods_queue = []
        self._current_period_idx = 0
        self._start_new_period()
    
    def _start_new_period(self):
        """Initialize a new sampling period."""
        self._current_period_data = {
            'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
            'action': torch.empty((0,), device=self.device, dtype=torch.long),
            'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
            'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),  # Will be resized on first add
            'alpha': torch.empty((0,), device=self.device, dtype=self.data_type),
        }
        self._trajectory_idx = np.array([], dtype=np.int32)
        self._unique_pairs = set()
        self._prev_obs = None
        self._prev_proprio = None
        self._traj_boundaries = {}
        self._current_dataset_idx = 0
        self._trajectory_active = False
    
    @property
    def data(self) -> Dict[str, torch.Tensor]:
        """
        Property for compatibility with existing code that accesses dataset.data.
        Returns aggregated data from all periods in the FIFO window plus current period.
        """
        if hasattr(self, '_sampled_data'):
            return self._sampled_data
        
        # Aggregate data from all periods in queue
        aggregated = self._aggregate_periods()
        
        
        # Add current period data
        if len(self._current_period_data['next_observation']) > 0:
            return {
                'observation': torch.cat([aggregated['observation'], self._current_period_data['observation']], dim=0),
                'action': torch.cat([aggregated['action'], self._current_period_data['action']], dim=0),
                'next_observation': torch.cat([aggregated['next_observation'], self._current_period_data['next_observation']], dim=0),
                'proprio_observation': torch.cat([aggregated['proprio_observation'], self._current_period_data['proprio_observation']], dim=0) if aggregated['proprio_observation'].shape[0] > 0 else self._current_period_data['proprio_observation'],
                'alpha': torch.cat([aggregated['alpha'], self._current_period_data['alpha']], dim=0)
            }
        else:
            return aggregated
    
    @property
    def current_data_size(self) -> int:
        """Size of current period data (number of transitions)."""
        return self.current_period_data_size
    
    @property
    def last_size(self) -> int:
        """Size of the last period that was added to the queue."""
        return self._last_period_size
    
    @property
    def size(self) -> int:
        """Total number of transitions across all periods in window (excluding current period)."""
        total = sum(len(period['data']['next_observation']) for period in self._periods_queue)
        return total
    
    @property
    def current_period_data_size(self) -> int:
        """Size of current period data."""
        if self._current_period_data is None:
            return 0
        return len(self._current_period_data['next_observation'])
    
    def add_pairs(self, state, action):
        """Track unique state-action pairs (for compatibility with ideal mode)."""
        pair = (np.argmax(state), action)
        self._unique_pairs.add(pair)
        return
    
    @property
    def is_complete(self) -> bool:
        """Check if current period has all unique state-action pairs."""
        return self.dataset_type == "unique" and len(self._unique_pairs) == self.expected_size
    
    @property
    def greater_equal_target_horizon(self) -> bool:
        """Check if current traj exceeds expected horizon size."""
        if not hasattr(self, 'current_target_horizon') or  len(self._traj_boundaries) == 0: #self._current_dataset_idx > len(self._traj_boundaries) or
            return False
        return (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon
    
    @property
    def reset_episode(self) -> bool:
        """Check if current traj exceeds expected horizon size."""
        if not hasattr(self, 'current_target_horizon') or  len(self._traj_boundaries) == 0: #self._current_dataset_idx > len(self._traj_boundaries) or
            return False
        if (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon+1:
            utils.ColorPrint.red("Resetting due to exceeding target horizon")
        return (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon+1
    
    def add_transition(self, time_step):
        """Add a transition to the current sampling period."""
        if self.dataset_type == "unique":
            self._add_unique(time_step)
        else:
            if self.dynamic_horizon== True:
                self._add_dynamic_horizon(time_step)
            else:
                self._add_all(time_step)
    
    def _add_unique(self, time_step):
        """Add only unique (s,a) pairs to current period."""
        if time_step.step_type == StepType.FIRST:
            self._prev_obs = time_step.observation
            self._prev_proprio = getattr(time_step, 'proprio_observation', None)
            self._current_dataset_idx += 1
            self._trajectory_active = True
            return
        
        if not self._trajectory_active:
            return
        
        if time_step.step_type in (StepType.MID, StepType.LAST):
            pair = (np.argmax(self._prev_obs), time_step.action)
            
            if pair not in self._unique_pairs:
                self._unique_pairs.add(pair)
                
                self._current_period_data['observation'] = torch.cat([
                    self._current_period_data['observation'],
                    torch.tensor(self._prev_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                self._current_period_data['action'] = torch.cat([
                    self._current_period_data['action'],
                    torch.tensor([time_step.action], device=self.device, dtype=torch.long)
                ], dim=0)
                self._current_period_data['next_observation'] = torch.cat([
                    self._current_period_data['next_observation'],
                    torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                
                # Add proprio observation if available
                if self._prev_proprio is not None:
                    proprio_tensor = torch.tensor(self._prev_proprio, device=self.device, dtype=self.data_type).unsqueeze(0)
                    if self._current_period_data['proprio_observation'].shape[0] == 0:
                        # Initialize with correct shape
                        self._current_period_data['proprio_observation'] = proprio_tensor
                    else:
                        self._current_period_data['proprio_observation'] = torch.cat([
                            self._current_period_data['proprio_observation'],
                            proprio_tensor
                        ], dim=0)
                
                alpha_val = 1.0 if len(self._unique_pairs) == 1 else 0.0
                self._current_period_data['alpha'] = torch.cat([
                    self._current_period_data['alpha'],
                    torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
                ], dim=0)
                
                self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
                
                current_idx = len(self._trajectory_idx) - 1
                if self._current_dataset_idx not in self._traj_boundaries:
                    self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
                else:
                    start_idx = self._traj_boundaries[self._current_dataset_idx][0]
                    self._traj_boundaries[self._current_dataset_idx] = (start_idx, current_idx)
                
                # Cache first complete transition for dummy
                self._cache_first_transition()
            
            self._prev_obs = time_step.observation
            self._prev_proprio = getattr(time_step, 'proprio_observation', None)
            if time_step.step_type == StepType.LAST:
                self._trajectory_active = False
    
    def _add_all(self, time_step):
        """Add all transitions to current period."""
        if time_step.step_type == StepType.FIRST:
            self._current_dataset_idx += 1
            self._trajectory_active = True
            current_idx = len(self._trajectory_idx)
            self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            alpha_val = 1.0 if self._current_period_data['observation'].shape[0] == 1 else 0.0
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
        
        elif time_step.step_type == StepType.MID:
            if not self._trajectory_active:
                return
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([0.0], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
            start_idx = self._traj_boundaries[self._current_dataset_idx][0]
            self._traj_boundaries[self._current_dataset_idx] = (start_idx, len(self._trajectory_idx) - 1)
            
            # Cache first complete transition for dummy
            self._cache_first_transition()
  
        elif time_step.step_type == StepType.LAST:
            if not self._trajectory_active:
                return
            
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._trajectory_active = False
    
    def _add_dynamic_horizon(self, time_step):
        """Add all transitions to current period."""
        if time_step.step_type == StepType.FIRST:
            # Horizon Computation

            prob = np.random.rand()
            horizon = np.log(1 - prob) / np.log(self.gamma) - 1
            self.current_target_horizon = int(np.round(horizon))
            
            # Track horizon for plotting
            if self.dynamic_horizon:
                self._horizon_history.append(self.current_target_horizon)
            
            ColorPrint.green(f"New trajectory with target horizon: {self.current_target_horizon}")
            # --------------------------------
            
            self._current_dataset_idx += 1
            self._trajectory_active = True
            current_idx = len(self._trajectory_idx)
            self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            alpha_val = 1.0 if self._current_period_data['observation'].shape[0] == 1 else 0.0
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
        
        elif time_step.step_type == StepType.MID:
            if not self._trajectory_active:
                return
            
            if not self.reset_episode:
                self._current_period_data['observation'] = torch.cat([
                    self._current_period_data['observation'],
                    torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                
                # Add proprio observation if available
                proprio_obs = getattr(time_step, 'proprio_observation', None)
                if proprio_obs is not None:
                    proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                    if self._current_period_data['proprio_observation'].shape[0] == 0:
                        self._current_period_data['proprio_observation'] = proprio_tensor
                    else:
                        self._current_period_data['proprio_observation'] = torch.cat([
                            self._current_period_data['proprio_observation'],
                            proprio_tensor
                        ], dim=0)
                
                self._current_period_data['alpha'] = torch.cat([
                    self._current_period_data['alpha'],
                    torch.tensor([0.0], device=self.device, dtype=self.data_type)
                ], dim=0)
            else:
                self._trajectory_active = False
            
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
            start_idx = self._traj_boundaries[self._current_dataset_idx][0]
            self._traj_boundaries[self._current_dataset_idx] = (start_idx, len(self._trajectory_idx) - 1)
            
            # Cache first complete transition for dummy
            self._cache_first_transition()
  
        elif time_step.step_type == StepType.LAST:
            if not self._trajectory_active:
                return
    
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._trajectory_active = False
    
    def _cache_first_transition(self):
        """Cache the first complete transition for use as dummy transition."""
        if self._dummy_cache is not None:
            return  # Already cached
        
        # Check if we have at least one complete transition
        if (len(self._current_period_data['observation']) > 0 and
            len(self._current_period_data['action']) > 0 and
            len(self._current_period_data['next_observation']) > 0):
            
            indices = torch.where(torch.all(self._current_period_data['next_observation'] == self.first_state, dim=1))[0]
            if indices.shape[0] == 0:
                return
            # indices = indices[0]

            self._dummy_cache = {
                'observation': self._current_period_data['observation'][indices:indices+1].clone(),
                'action': self._current_period_data['action'][indices:indices+1].clone(),
                'next_observation': self._current_period_data['next_observation'][indices:indices+1].clone(),
                'alpha': self._current_period_data['alpha'][indices:indices+1].clone()
            }
            
            # Add proprio if available
            if self._current_period_data['proprio_observation'].shape[0] > 0:
                self._dummy_cache['proprio_observation'] = self._current_period_data['proprio_observation'][indices:indices+1].clone()
            else:
                self._dummy_cache['proprio_observation'] = torch.empty((1, 0), device=self.device, dtype=self.data_type)
            
            utils.ColorPrint.green("Cached first transition for dummy use")
    
    def add_dummy_transition(self):
        """Add a dummy transition using the cached first sample."""
        if self._dummy_cache is None:
            utils.ColorPrint.yellow("No dummy cache available, skipping dummy transition")
            return
        
        self._current_period_data['observation'] = torch.cat([
            self._dummy_cache['observation'], 
            self._current_period_data['observation']
        ], dim=0)
        self._current_period_data['action'] = torch.cat([
            self._dummy_cache['action'], 
            self._current_period_data['action']
        ], dim=0)
        self._current_period_data['next_observation'] = torch.cat([
            self._dummy_cache['next_observation'], 
            self._current_period_data['next_observation']
        ], dim=0)
        self._current_period_data['alpha'] = torch.cat([
            self._dummy_cache['alpha'], 
            self._current_period_data['alpha']
        ], dim=0)
        
        # Add proprio if available
        if self._dummy_cache['proprio_observation'].shape[0] > 0:
            if 'proprio_observation' not in self._current_period_data or self._current_period_data['proprio_observation'].shape[0] == 0:
                self._current_period_data['proprio_observation'] = self._dummy_cache['proprio_observation']
            else:
                self._current_period_data['proprio_observation'] = torch.cat([
                    self._dummy_cache['proprio_observation'],
                    self._current_period_data['proprio_observation']
                ], dim=0)
    
    def _aggregate_periods(self) -> Dict[str, torch.Tensor]:
        """Concatenate data from all periods in the window."""
        if len(self._periods_queue) == 0:
            return {
                'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'action': torch.empty((0,), device=self.device, dtype=torch.long),
                'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),
                'alpha': torch.empty((0,), device=self.device, dtype=self.data_type)
            }
        
        all_obs = []
        all_actions = []
        all_next_obs = []
        all_proprio = []
        all_alpha = []
        
        for period in self._periods_queue:
            data = period['data']
            if len(data['next_observation']) > 0:
                all_obs.append(data['observation'])
                all_actions.append(data['action'])
                all_next_obs.append(data['next_observation'])
                if data['proprio_observation'].shape[0] > 0:
                    all_proprio.append(data['proprio_observation'])
                all_alpha.append(data['alpha'])
        
        if len(all_obs) == 0:
            return {
                'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'action': torch.empty((0,), device=self.device, dtype=torch.long),
                'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),
                'alpha': torch.empty((0,), device=self.device, dtype=self.data_type)
            }
        
        return {
            'observation': torch.cat(all_obs, dim=0),
            'action': torch.cat(all_actions, dim=0),
            'next_observation': torch.cat(all_next_obs, dim=0),
            'proprio_observation': torch.cat(all_proprio, dim=0) if len(all_proprio) > 0 else torch.empty((0, 0), device=self.device, dtype=self.data_type),
            'alpha': torch.cat(all_alpha, dim=0)
        }
    
    def get_data(self, unique=False) -> Dict[str, torch.Tensor]:
        """
        End current sampling period, clean incomplete trajectories, add to FIFO queue, 
        maintain window size, and return aggregated data from last N periods.
        
        Returns:
            Dictionary with concatenated data from all periods in window
        """
        # Plot horizon histogram before resetting
        if self.dynamic_horizon:
            self._plot_horizon_histogram()
        else:
            # Clean incomplete trajectories from current period
            self._clean_incomplete_trajectories()
        
        self.add_dummy_transition()
        # Store current period data with metadata
        period_entry = {
            'data': deepcopy(self._current_period_data),
            'trajectory_idx': self._trajectory_idx.copy(),
            'traj_boundaries': deepcopy(self._traj_boundaries),
            'period_idx': self._current_period_idx
        }
        
        # Track the size of this period
        self._last_period_size = len(self._current_period_data['next_observation'])
        
        # Add to queue
        self._periods_queue.append(period_entry)
        utils.ColorPrint.green(f"Completed sampling period {self._current_period_idx} with {self._last_period_size} transitions.")
        
        # Maintain FIFO: remove oldest if exceeds window size
        if len(self._periods_queue) > self.window_size:
            removed = self._periods_queue.pop(0)
            utils.ColorPrint.yellow(f"Removed oldest period {removed['period_idx']} from FIFO queue.")
        
        # Aggregate data from all periods in window
        aggregated_data = self._aggregate_periods()
        
        # Start new period
        self._current_period_idx += 1
        self._start_new_period()
        
        # Reset horizon history after plotting
        if self.dynamic_horizon:
            self._horizon_history = []
        
        # Filter for unique state-action pairs if requested
        if unique and len(aggregated_data['next_observation']) > 0:
            aggregated_data = self._filter_unique_state_action_pairs(aggregated_data)
        
        # Apply subsampling if needed
        if self.n_subsamples is not None and len(aggregated_data['next_observation']) > 0:
            assert unique is False, "Subsampling with unique state-action pairs is not supported."
            if self.subsampling_strategy == "random":
                aggregated_data = self._subsample_data(aggregated_data)
            elif self.subsampling_strategy == "eder":
                aggregated_data = self._eder_subsampling(aggregated_data)
        
        self._sampled_data = aggregated_data
        
        return aggregated_data

    
    def _filter_unique_state_action_pairs(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter data to keep only unique state-action pairs.
        For duplicate (s,a) pairs, keeps the first occurrence.
        
        Args:
            data: Dictionary with observation, action, next_observation, alpha tensors
            
        Returns:
            Filtered dictionary with only unique (s,a) pairs
        """
        if len(data['observation']) == 0:
            return data
        
        # Convert observations to state indices (assuming one-hot encoding)
        state_indices = torch.argmax(data['observation'], dim=1).cpu().numpy()
        action_indices = data['action'].cpu().numpy()
        
        # Track unique (state, action) pairs and their first occurrence indices
        seen_pairs = set()
        unique_indices = []
        
        for idx in range(len(state_indices)):
            pair = (int(state_indices[idx]), int(action_indices[idx]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_indices.append(idx)
        
        utils.ColorPrint.green(f"Filtered {len(state_indices)} transitions to {len(unique_indices)} unique state-action pairs.")
        
        # Return filtered data
        return {
            'observation': data['observation'][unique_indices],
            'action': data['action'][unique_indices],
            'next_observation': data['next_observation'][unique_indices],
            'alpha': data['alpha'][unique_indices],
            'proprio_observation': data['proprio_observation'][unique_indices] if 'proprio_observation' in data else torch.empty((0, 0), device=self.device, dtype=self.data_type)
        }
    
    def _clean_incomplete_trajectories(self):
        """Remove data from incomplete trajectories in the current period."""
        if len(self._traj_boundaries) == 0:
            return
        
        # Find incomplete trajectories (where trajectory is still active)
        incomplete_traj_ids = []
        for traj_id in self._traj_boundaries.keys():
            # Check if this trajectory is still active (not properly terminated)
            if traj_id == self._current_dataset_idx and self._trajectory_active:
                incomplete_traj_ids.append(traj_id)
        
        if len(incomplete_traj_ids) == 0:
            return
        
        utils.ColorPrint.yellow(f"Cleaning {len(incomplete_traj_ids)} incomplete trajectory/trajectories from current period.")
        
        # Find indices to keep (all indices not belonging to incomplete trajectories)
        indices_to_keep = []
        for idx, traj_id in enumerate(self._trajectory_idx):
            if traj_id not in incomplete_traj_ids:
                indices_to_keep.append(idx)
        
        if len(indices_to_keep) == 0:
            # All data was from incomplete trajectories, reset current period
            utils.ColorPrint.yellow("All data in current period was from incomplete trajectories. Resetting period.")
            self._start_new_period()
            return
        
        # Filter data to keep only complete trajectories
        self._current_period_data['observation'] = self._current_period_data['observation'][indices_to_keep]
        self._current_period_data['alpha'] = self._current_period_data['alpha'][indices_to_keep]
        
        # For action and next_observation, we need to handle the offset
        # These arrays may have one less element than observation
        action_indices = [i for i in indices_to_keep if i < len(self._current_period_data['action'])]
        self._current_period_data['action'] = self._current_period_data['action'][action_indices]
        self._current_period_data['next_observation'] = self._current_period_data['next_observation'][action_indices]
        
        # Update trajectory_idx
        self._trajectory_idx = self._trajectory_idx[indices_to_keep]
        
        # Remove incomplete trajectories from boundaries
        for traj_id in incomplete_traj_ids:
            del self._traj_boundaries[traj_id]
        
        # Rebuild trajectory boundaries with new indices
        new_boundaries = {}
        for traj_id in self._traj_boundaries.keys():
            # Find min and max indices for this trajectory in the filtered data
            traj_indices = [i for i, tid in enumerate(self._trajectory_idx) if tid == traj_id]
            if len(traj_indices) > 0:
                new_boundaries[traj_id] = (min(traj_indices), max(traj_indices))
        
        self._traj_boundaries = new_boundaries
        utils.ColorPrint.green(f"Cleaned period now has {len(self._current_period_data['observation'])} observations.")
    
    def _subsample_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Subsample data using geometric distribution based on gamma."""
        total_size = len(data['next_observation'])
        
        if self.n_subsamples >= total_size:
            utils.ColorPrint.yellow(f"Requested subsample size {self.n_subsamples} >= total size {total_size}. Returning full data.")
            return data
        
        utils.ColorPrint.green(f"Subsampling {self.n_subsamples} from {total_size} transitions.")
        
        # Random subsampling (can be enhanced with trajectory-aware sampling)
        indices = np.random.choice(total_size, size=self.n_subsamples, replace=False)
        
        return {
            'observation': data['observation'][indices],
            'action': data['action'][indices],
            'next_observation': data['next_observation'][indices],
            'alpha': data['alpha'][indices],
            'proprio_observation': data['proprio_observation'][indices] if 'proprio_observation' in data else torch.emp
        }

    def spd_logdet_cholesky(self, K, jitter=1e-6):
        # K: (..., n, n), symmetric PSD/SPD kernel submatrix
        # K = 0.5 * (K + K.transpose(-1, -2))
        n = self.n_subsamples
        I = torch.eye(n, device=K.device, dtype=K.dtype)

        L, info = torch.linalg.cholesky_ex(K + jitter * I, upper=False, check_errors=False)
        if torch.any(info != 0):
            raise RuntimeError("Cholesky failed; increase jitter or check kernel definiteness.")
        d = L.diagonal(dim1=-2, dim2=-1)
        return 2.0 * d.log().sum(dim=-1)

    def _eder_subsampling(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Subsample data using EDER method (not implemented)."""
        
        total_size = len(data['next_observation'])
        
        
        if self.n_subsamples >= total_size:
            utils.ColorPrint.yellow(f"Requested subsample size {self.n_subsamples} >= total size {total_size}. Returning full data.")
            return data
        
        utils.ColorPrint.green(f"Subsampling {self.n_subsamples} from {total_size} transitions.")
        
        starting_search_time = time()

        tmp_max = -np.inf

        for i in range(100):
            # Random subsampling (can be enhanced with trajectory-aware sampling)
            indices = np.random.choice(total_size, size=self.n_subsamples, replace=False)
            
            sampled_data = {
                'observation': data['observation'][indices],
                'action': data['action'][indices],
                'next_observation': data['next_observation'][indices],
                'proprio_observation': data['proprio_observation'][indices] if 'proprio_observation' in data else torch.empty((self.n_subsamples, 0), device=self.device, dtype=self.data_type),
                'alpha': data['alpha'][indices]
            }

            # Encoding state-action pairs as unique integers
            action_onehot = F.one_hot(sampled_data['action'].long(), self.n_actions).reshape(-1, self.n_actions)  # [B, |A|]
            
            # Outer product: [B, d] ⊗ [B, |A|] -> [B, d*|A|]
            encoded_sa = torch.einsum('bd,ba->bda', sampled_data['observation'], action_onehot).reshape(self.n_subsamples, -1)

            kernel_sa = encoded_sa@encoded_sa.T # [B, B]

            log_det =self.spd_logdet_cholesky(kernel_sa)

            # if np.random.rand() <= np.exp(log_det - self.max_log_det):
            # if log_det > self.max_log_det:

            #     if self.max_log_det == -np.inf:
            #         ColorPrint.green(f"EDER subsampling with log-det: {log_det.item():.4f}  in {time() - starting_search_time:.2f}s accepted after {i+1} attempts")
            #     else:
            #         ColorPrint.green(f"EDER subsampling with log-det: {log_det.item():.4f} (max: {self.max_log_det:.4f}) in {time() - starting_search_time:.2f}s accepted after {i+1} attempts")
            #     if log_det > self.max_log_det:
            #         self.max_log_det = log_det.item()
            #     return sampled_data
            
            if log_det > tmp_max:
                tmp_max = log_det.item()
                best_sampled_data = sampled_data


            if log_det > self.max_log_det:
                self.max_log_det = log_det.item()
        ColorPrint.red(f"EDER subsampling failed to find better subset; returning best sampled subset, with log-det: {tmp_max:.4f} (max: {self.max_log_det:.4f}) in {time() - starting_search_time:.2f}s")
        return best_sampled_data
    
    def _plot_horizon_histogram(self):
        """Plot histogram of dynamic horizons and save to file."""
        if not self.dynamic_horizon or len(self._horizon_history) == 0:
            return
        
        save_dir = os.path.join(os.getcwd(), "horizon_plots")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"horizon_histogram_period_{self._plot_counter}.png")
        
        # Get range of horizons
        min_horizon = min(self._horizon_history)
        max_horizon = max(self._horizon_history)
        
        # Create bins that include all integer values from min to max
        bins = np.arange(min_horizon - 0.5, max_horizon + 1.5, 1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(self._horizon_history, bins=bins, edgecolor='black', alpha=0.7, align='mid')
        
        # Set x-axis ticks to show each integer horizon value
        plt.xticks(range(min_horizon, max_horizon + 1))
        
        plt.xlabel('Target Horizon')
        plt.ylabel('Frequency')
        plt.title(f'Dynamic Horizon Distribution - Period {self._plot_counter}\n'
                  f'Total trajectories: {len(self._horizon_history)}, '
                  f'Mean: {np.mean(self._horizon_history):.2f}, '
                  f'Std: {np.std(self._horizon_history):.2f}')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        ColorPrint.green(f"Saved horizon histogram to {save_path}")
        
        # Increment counter for next plot
        self._plot_counter += 1



class InternalDatasetFIFOV2:
    """
    FIFO-based internal dataset that maintains only the last N sampling periods.
    Each call to get_data() marks the end of a sampling period and retrieves
    data from the last N periods.
    """
    
    def __init__(self, dataset_type: str, n_states: int, n_actions: int, 
                 gamma: float, window_size: int, n_subsamples: int, 
                 subsampling_strategy: str, dynamic_horizon: bool = False,
                 obs_shape: tuple = None,
                 device: str = 'cpu', data_type=torch.double, first_state = None, second_state = None):
        """
        Args:
            dataset_type: "unique" or "all"
            n_states: Number of states
            n_actions: Number of actions
            gamma: Discount factor for geometric sampling
            window_size: Number of sampling periods to keep in memory
            n_subsamples: Number of samples to return per period (None = all)
            subsampling_strategy: Strategy for subsampling ("random" or "eder")
            dynamic_horizon: Whether to use a dynamic horizon
            obs_shape: Shape of observations (e.g., (84, 84, 3) for images)
            device: Torch device
        """
        self.dataset_type = dataset_type
        self.n_states = n_states
        self.n_actions = n_actions
        self.expected_size = n_states * n_actions
        assert dataset_type in ("unique", "all"), "dataset_type must be 'unique' or 'all'"
        self.n_subsamples = n_subsamples
        self.gamma = gamma
        self.window_size = window_size
        self.device = torch.device(device)
        self.dynamic_horizon = dynamic_horizon
        self.obs_shape = obs_shape if obs_shape is not None else (n_states,)
        self.data_type = data_type
        self.first_state = first_state
        self.second_state = second_state
        
        # FIFO queue: list of sampling periods, each period is a dict of tensors
        self._periods_queue = []
        self._current_period_data = None
        self._current_period_idx = 0
        self._last_period_size = 0  # Track size of last added period
        self.max_log_det = -np.inf
        self.subsampling_strategy = subsampling_strategy
        
        # Track horizons for dynamic horizon mode
        self._horizon_history = []
        self._plot_counter = 0
        
        # Cache for dummy transition (first complete sample)
        self._dummy_cache = None
        
        self.reset()
    
    def reset(self):
        """Reset the FIFO dataset."""
        utils.ColorPrint.yellow("Resetting FIFO internal dataset.")
        self._periods_queue = []
        self._current_period_idx = 0
        self._start_new_period()
    
    def _start_new_period(self):
        """Initialize a new sampling period."""
        self._current_period_data = {
            'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
            'action': torch.empty((0,), device=self.device, dtype=torch.long),
            'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
            'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),  # Will be resized on first add
            'alpha': torch.empty((0,), device=self.device, dtype=self.data_type),
        }
        self._trajectory_idx = np.array([], dtype=np.int32)
        self._unique_pairs = set()
        self._prev_obs = None
        self._prev_proprio = None
        self._traj_boundaries = {}
        self._current_dataset_idx = 0
        self._trajectory_active = False
    
    @property
    def data(self) -> Dict[str, torch.Tensor]:
        """
        Property for compatibility with existing code that accesses dataset.data.
        Returns aggregated data from all periods in the FIFO window plus current period.
        """
        if hasattr(self, '_sampled_data'):
            return self._sampled_data
        
        # Aggregate data from all periods in queue
        aggregated = self._aggregate_periods()
        
        
        # Add current period data
        if len(self._current_period_data['next_observation']) > 0:
            return {
                'observation': torch.cat([aggregated['observation'], self._current_period_data['observation']], dim=0),
                'action': torch.cat([aggregated['action'], self._current_period_data['action']], dim=0),
                'next_observation': torch.cat([aggregated['next_observation'], self._current_period_data['next_observation']], dim=0),
                'proprio_observation': torch.cat([aggregated['proprio_observation'], self._current_period_data['proprio_observation']], dim=0) if aggregated['proprio_observation'].shape[0] > 0 else self._current_period_data['proprio_observation'],
                'alpha': torch.cat([aggregated['alpha'], self._current_period_data['alpha']], dim=0)
            }
        else:
            return aggregated
    
    @property
    def current_data_size(self) -> int:
        """Size of current period data (number of transitions)."""
        return self.current_period_data_size
    
    @property
    def last_size(self) -> int:
        """Size of the last period that was added to the queue."""
        return self._last_period_size
    
    @property
    def size(self) -> int:
        """Total number of transitions across all periods in window (excluding current period)."""
        total = sum(len(period['data']['next_observation']) for period in self._periods_queue)
        return total
    
    @property
    def current_period_data_size(self) -> int:
        """Size of current period data."""
        if self._current_period_data is None:
            return 0
        return len(self._current_period_data['next_observation'])
    
    def add_pairs(self, state, action):
        """Track unique state-action pairs (for compatibility with ideal mode)."""
        pair = (np.argmax(state), action)
        self._unique_pairs.add(pair)
        return
    
    @property
    def is_complete(self) -> bool:
        """Check if current period has all unique state-action pairs."""
        return self.dataset_type == "unique" and len(self._unique_pairs) == self.expected_size
    
    @property
    def greater_equal_target_horizon(self) -> bool:
        """Check if current traj exceeds expected horizon size."""
        if not hasattr(self, 'current_target_horizon') or  len(self._traj_boundaries) == 0: #self._current_dataset_idx > len(self._traj_boundaries) or
            return False
        return (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon
    
    @property
    def reset_episode(self) -> bool:
        """Check if current traj exceeds expected horizon size."""
        if not hasattr(self, 'current_target_horizon') or  len(self._traj_boundaries) == 0: #self._current_dataset_idx > len(self._traj_boundaries) or
            return False
        if (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon+1:
            utils.ColorPrint.red("Resetting due to exceeding target horizon")
        return (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon+1
    
    def add_transition(self, time_step):
        """Add a transition to the current sampling period."""
        if self.dataset_type == "unique":
            self._add_unique(time_step)
        else:
            if self.dynamic_horizon== True:
                self._add_dynamic_horizon(time_step)
            else:
                self._add_all(time_step)
    
    def _add_unique(self, time_step):
        """Add only unique (s,a) pairs to current period."""
        if time_step.step_type == StepType.FIRST:
            self._prev_obs = time_step.observation
            self._prev_proprio = getattr(time_step, 'proprio_observation', None)
            self._current_dataset_idx += 1
            self._trajectory_active = True
            return
        
        if not self._trajectory_active:
            return
        
        if time_step.step_type in (StepType.MID, StepType.LAST):
            pair = (np.argmax(self._prev_obs), time_step.action)
            
            if pair not in self._unique_pairs:
                self._unique_pairs.add(pair)
                
                self._current_period_data['observation'] = torch.cat([
                    self._current_period_data['observation'],
                    torch.tensor(self._prev_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                self._current_period_data['action'] = torch.cat([
                    self._current_period_data['action'],
                    torch.tensor([time_step.action], device=self.device, dtype=torch.long)
                ], dim=0)
                self._current_period_data['next_observation'] = torch.cat([
                    self._current_period_data['next_observation'],
                    torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                
                # Add proprio observation if available
                if self._prev_proprio is not None:
                    proprio_tensor = torch.tensor(self._prev_proprio, device=self.device, dtype=self.data_type).unsqueeze(0)
                    if self._current_period_data['proprio_observation'].shape[0] == 0:
                        # Initialize with correct shape
                        self._current_period_data['proprio_observation'] = proprio_tensor
                    else:
                        self._current_period_data['proprio_observation'] = torch.cat([
                            self._current_period_data['proprio_observation'],
                            proprio_tensor
                        ], dim=0)
                
                alpha_val = 1.0 if len(self._unique_pairs) == 1 else 0.0
                self._current_period_data['alpha'] = torch.cat([
                    self._current_period_data['alpha'],
                    torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
                ], dim=0)
                
                self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
                
                current_idx = len(self._trajectory_idx) - 1
                if self._current_dataset_idx not in self._traj_boundaries:
                    self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
                else:
                    start_idx = self._traj_boundaries[self._current_dataset_idx][0]
                    self._traj_boundaries[self._current_dataset_idx] = (start_idx, current_idx)
                
                # Cache first complete transition for dummy
                self._cache_first_transition()
            
            self._prev_obs = time_step.observation
            self._prev_proprio = getattr(time_step, 'proprio_observation', None)
            if time_step.step_type == StepType.LAST:
                self._trajectory_active = False
    
    def _add_all(self, time_step):
        """Add all transitions to current period."""
        if time_step.step_type == StepType.FIRST:
            self._current_dataset_idx += 1
            self._trajectory_active = True
            current_idx = len(self._trajectory_idx)
            self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            alpha_val = 1.0 if self._current_period_data['observation'].shape[0] == 1 else 0.0
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
        
        elif time_step.step_type == StepType.MID:
            if not self._trajectory_active:
                return
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([0.0], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
            start_idx = self._traj_boundaries[self._current_dataset_idx][0]
            self._traj_boundaries[self._current_dataset_idx] = (start_idx, len(self._trajectory_idx) - 1)
            
            # Cache first complete transition for dummy
            self._cache_first_transition()
  
        elif time_step.step_type == StepType.LAST:
            if not self._trajectory_active:
                return
            
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._trajectory_active = False
    
    def _add_dynamic_horizon(self, time_step):
        """Add all transitions to current period."""
        if time_step.step_type == StepType.FIRST:
            # Horizon Computation

            prob = np.random.rand()
            horizon = np.log(1 - prob) / np.log(self.gamma) - 1
            self.current_target_horizon = int(np.round(horizon))
            
            # Track horizon for plotting
            if self.dynamic_horizon:
                self._horizon_history.append(self.current_target_horizon)
            
            ColorPrint.green(f"New trajectory with target horizon: {self.current_target_horizon}")
            # --------------------------------
            
            self._current_dataset_idx += 1
            self._trajectory_active = True
            current_idx = len(self._trajectory_idx)
            self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            alpha_val = 1.0 if self._current_period_data['observation'].shape[0] == 1 else 0.0
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
        
        elif time_step.step_type == StepType.MID:
            if not self._trajectory_active:
                return
            
            if not self.reset_episode:
                self._current_period_data['observation'] = torch.cat([
                    self._current_period_data['observation'],
                    torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                
                # Add proprio observation if available
                proprio_obs = getattr(time_step, 'proprio_observation', None)
                if proprio_obs is not None:
                    proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                    if self._current_period_data['proprio_observation'].shape[0] == 0:
                        self._current_period_data['proprio_observation'] = proprio_tensor
                    else:
                        self._current_period_data['proprio_observation'] = torch.cat([
                            self._current_period_data['proprio_observation'],
                            proprio_tensor
                        ], dim=0)
                
                self._current_period_data['alpha'] = torch.cat([
                    self._current_period_data['alpha'],
                    torch.tensor([0.0], device=self.device, dtype=self.data_type)
                ], dim=0)
            else:
                self._trajectory_active = False
            
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
            start_idx = self._traj_boundaries[self._current_dataset_idx][0]
            self._traj_boundaries[self._current_dataset_idx] = (start_idx, len(self._trajectory_idx) - 1)
            
            # Cache first complete transition for dummy
            self._cache_first_transition()
  
        elif time_step.step_type == StepType.LAST:
            if not self._trajectory_active:
                return
    
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._trajectory_active = False
    
    def _cache_first_transition(self):
        """Cache the first complete transition for use as dummy transition."""
        if self._dummy_cache is not None:
            return  # Already cached
        
        # Check if we have at least one complete transition
        if (len(self._current_period_data['observation']) > 0 and
            len(self._current_period_data['action']) > 0 and
            len(self._current_period_data['next_observation']) > 0):
            
            indices = torch.where(torch.all(self._current_period_data['next_observation'] == self.first_state, dim=1))[0]
            if indices.shape[0] == 0:
                return
            # indices = indices[0]

            self._dummy_cache = {
                'observation': self._current_period_data['observation'][indices:indices+1].clone(),
                'action': self._current_period_data['action'][indices:indices+1].clone(),
                'next_observation': self._current_period_data['next_observation'][indices:indices+1].clone(),
                'alpha': self._current_period_data['alpha'][indices:indices+1].clone()
            }
            
            # Add proprio if available
            if self._current_period_data['proprio_observation'].shape[0] > 0:
                self._dummy_cache['proprio_observation'] = self._current_period_data['proprio_observation'][indices:indices+1].clone()
            else:
                self._dummy_cache['proprio_observation'] = torch.empty((1, 0), device=self.device, dtype=self.data_type)
            
            utils.ColorPrint.green("Cached first transition for dummy use")
    
    def add_dummy_transition(self):
        """Add a dummy transition using the cached first sample."""
        if self._dummy_cache is None:
            utils.ColorPrint.yellow("No dummy cache available, skipping dummy transition")
            return
        
        self._current_period_data['observation'] = torch.cat([
            self._dummy_cache['observation'], 
            self._current_period_data['observation']
        ], dim=0)
        self._current_period_data['action'] = torch.cat([
            self._dummy_cache['action'], 
            self._current_period_data['action']
        ], dim=0)
        self._current_period_data['next_observation'] = torch.cat([
            self._dummy_cache['next_observation'], 
            self._current_period_data['next_observation']
        ], dim=0)
        self._current_period_data['alpha'] = torch.cat([
            self._dummy_cache['alpha'], 
            self._current_period_data['alpha']
        ], dim=0)
        
        # Add proprio if available
        if self._dummy_cache['proprio_observation'].shape[0] > 0:
            if 'proprio_observation' not in self._current_period_data or self._current_period_data['proprio_observation'].shape[0] == 0:
                self._current_period_data['proprio_observation'] = self._dummy_cache['proprio_observation']
            else:
                self._current_period_data['proprio_observation'] = torch.cat([
                    self._dummy_cache['proprio_observation'],
                    self._current_period_data['proprio_observation']
                ], dim=0)
    
    def _aggregate_periods(self) -> Dict[str, torch.Tensor]:
        """Concatenate data from all periods in the window."""
        if len(self._periods_queue) == 0:
            return {
                'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'action': torch.empty((0,), device=self.device, dtype=torch.long),
                'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),
                'alpha': torch.empty((0,), device=self.device, dtype=self.data_type)
            }
        
        all_obs = []
        all_actions = []
        all_next_obs = []
        all_proprio = []
        all_alpha = []
        
        for period in self._periods_queue:
            data = period['data']
            if len(data['next_observation']) > 0:
                all_obs.append(data['observation'])
                all_actions.append(data['action'])
                all_next_obs.append(data['next_observation'])
                if data['proprio_observation'].shape[0] > 0:
                    all_proprio.append(data['proprio_observation'])
                all_alpha.append(data['alpha'])
        
        if len(all_obs) == 0:
            return {
                'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'action': torch.empty((0,), device=self.device, dtype=torch.long),
                'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),
                'alpha': torch.empty((0,), device=self.device, dtype=self.data_type)
            }
        
        return {
            'observation': torch.cat(all_obs, dim=0),
            'action': torch.cat(all_actions, dim=0),
            'next_observation': torch.cat(all_next_obs, dim=0),
            'proprio_observation': torch.cat(all_proprio, dim=0) if len(all_proprio) > 0 else torch.empty((0, 0), device=self.device, dtype=self.data_type),
            'alpha': torch.cat(all_alpha, dim=0)
        }
    
    def get_data(self, unique=False) -> Dict[str, torch.Tensor]:
        """
        End current sampling period, clean incomplete trajectories, add to FIFO queue, 
        maintain window size, and return aggregated data from last N periods.
        
        Returns:
            Dictionary with concatenated data from all periods in window
        """
        # Plot horizon histogram before resetting
        if self.dynamic_horizon:
            self._plot_horizon_histogram()
        else:
            # Clean incomplete trajectories from current period
            self._clean_incomplete_trajectories()
        
        self.add_dummy_transition()
        # Store current period data with metadata
        period_entry = {
            'data': deepcopy(self._current_period_data),
            'trajectory_idx': self._trajectory_idx.copy(),
            'traj_boundaries': deepcopy(self._traj_boundaries),
            'period_idx': self._current_period_idx
        }
        
        # Track the size of this period
        self._last_period_size = len(self._current_period_data['next_observation'])
        
        # Add to queue
        self._periods_queue.append(period_entry)
        utils.ColorPrint.green(f"Completed sampling period {self._current_period_idx} with {self._last_period_size} transitions.")
        
        # Maintain FIFO: remove oldest if exceeds window size
        if len(self._periods_queue) > self.window_size:
            removed = self._periods_queue.pop(0)
            utils.ColorPrint.yellow(f"Removed oldest period {removed['period_idx']} from FIFO queue.")
        
        # Aggregate data from all periods in window
        aggregated_data = self._aggregate_periods()
        
        # Start new period
        self._current_period_idx += 1
        self._start_new_period()
        
        # Reset horizon history after plotting
        if self.dynamic_horizon:
            self._horizon_history = []
        
        # Filter for unique state-action pairs if requested
        if unique and len(aggregated_data['next_observation']) > 0:
            aggregated_data = self._filter_unique_state_action_pairs(aggregated_data)
        
        # Apply subsampling if needed
        if self.n_subsamples is not None and len(aggregated_data['next_observation']) > 0:
            assert unique is False, "Subsampling with unique state-action pairs is not supported."
            if self.subsampling_strategy == "random":
                aggregated_data = self._subsample_data(aggregated_data)
            elif self.subsampling_strategy == "eder":
                aggregated_data = self._eder_subsampling(aggregated_data)
        
        self._sampled_data = aggregated_data
        
        return aggregated_data

    
    def _filter_unique_state_action_pairs(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter data to keep only unique state-action pairs.
        For duplicate (s,a) pairs, keeps the first occurrence.
        
        Args:
            data: Dictionary with observation, action, next_observation, alpha tensors
            
        Returns:
            Filtered dictionary with only unique (s,a) pairs
        """
        if len(data['observation']) == 0:
            return data
        
        # Convert observations to state indices (assuming one-hot encoding)
        state_indices = torch.argmax(data['observation'], dim=1).cpu().numpy()
        action_indices = data['action'].cpu().numpy()
        
        # Track unique (state, action) pairs and their first occurrence indices
        seen_pairs = set()
        unique_indices = []
        
        for idx in range(len(state_indices)):
            pair = (int(state_indices[idx]), int(action_indices[idx]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_indices.append(idx)
        
        utils.ColorPrint.green(f"Filtered {len(state_indices)} transitions to {len(unique_indices)} unique state-action pairs.")
        
        # Return filtered data
        return {
            'observation': data['observation'][unique_indices],
            'action': data['action'][unique_indices],
            'next_observation': data['next_observation'][unique_indices],
            'alpha': data['alpha'][unique_indices],
            'proprio_observation': data['proprio_observation'][unique_indices] if 'proprio_observation' in data else torch.empty((0, 0), device=self.device, dtype=self.data_type)
        }
    
    def _clean_incomplete_trajectories(self):
        """Remove data from incomplete trajectories in the current period."""
        if len(self._traj_boundaries) == 0:
            return
        
        # Find incomplete trajectories (where trajectory is still active)
        incomplete_traj_ids = []
        for traj_id in self._traj_boundaries.keys():
            # Check if this trajectory is still active (not properly terminated)
            if traj_id == self._current_dataset_idx and self._trajectory_active:
                incomplete_traj_ids.append(traj_id)
        
        if len(incomplete_traj_ids) == 0:
            return
        
        utils.ColorPrint.yellow(f"Cleaning {len(incomplete_traj_ids)} incomplete trajectory/trajectories from current period.")
        
        # Find indices to keep (all indices not belonging to incomplete trajectories)
        indices_to_keep = []
        for idx, traj_id in enumerate(self._trajectory_idx):
            if traj_id not in incomplete_traj_ids:
                indices_to_keep.append(idx)
        
        if len(indices_to_keep) == 0:
            # All data was from incomplete trajectories, reset current period
            utils.ColorPrint.yellow("All data in current period was from incomplete trajectories. Resetting period.")
            self._start_new_period()
            return
        
        # Filter data to keep only complete trajectories
        self._current_period_data['observation'] = self._current_period_data['observation'][indices_to_keep]
        self._current_period_data['alpha'] = self._current_period_data['alpha'][indices_to_keep]
        
        # For action and next_observation, we need to handle the offset
        # These arrays may have one less element than observation
        action_indices = [i for i in indices_to_keep if i < len(self._current_period_data['action'])]
        self._current_period_data['action'] = self._current_period_data['action'][action_indices]
        self._current_period_data['next_observation'] = self._current_period_data['next_observation'][action_indices]
        
        # Update trajectory_idx
        self._trajectory_idx = self._trajectory_idx[indices_to_keep]
        
        # Remove incomplete trajectories from boundaries
        for traj_id in incomplete_traj_ids:
            del self._traj_boundaries[traj_id]
        
        # Rebuild trajectory boundaries with new indices
        new_boundaries = {}
        for traj_id in self._traj_boundaries.keys():
            # Find min and max indices for this trajectory in the filtered data
            traj_indices = [i for i, tid in enumerate(self._trajectory_idx) if tid == traj_id]
            if len(traj_indices) > 0:
                new_boundaries[traj_id] = (min(traj_indices), max(traj_indices))
        
        self._traj_boundaries = new_boundaries
        utils.ColorPrint.green(f"Cleaned period now has {len(self._current_period_data['observation'])} observations.")
    
    def _subsample_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Subsample data using geometric distribution based on gamma."""
        total_size = len(data['next_observation'])
        
        if self.n_subsamples >= total_size:
            utils.ColorPrint.yellow(f"Requested subsample size {self.n_subsamples} >= total size {total_size}. Returning full data.")
            return data
        
        utils.ColorPrint.green(f"Subsampling {self.n_subsamples} from {total_size} transitions.")
        
        # Random subsampling (can be enhanced with trajectory-aware sampling)
        indices = np.random.choice(total_size, size=self.n_subsamples, replace=False)
        
        return {
            'observation': data['observation'][indices],
            'action': data['action'][indices],
            'next_observation': data['next_observation'][indices],
            'alpha': data['alpha'][indices],
            'proprio_observation': data['proprio_observation'][indices] if 'proprio_observation' in data else torch.emp
        }

    def spd_logdet_cholesky(self, K, jitter=1e-6):
        # K: (..., n, n), symmetric PSD/SPD kernel submatrix
        # K = 0.5 * (K + K.transpose(-1, -2))
        n = self.n_subsamples
        I = torch.eye(n, device=K.device, dtype=K.dtype)

        L, info = torch.linalg.cholesky_ex(K + jitter * I, upper=False, check_errors=False)
        if torch.any(info != 0):
            raise RuntimeError("Cholesky failed; increase jitter or check kernel definiteness.")
        d = L.diagonal(dim1=-2, dim2=-1)
        return 2.0 * d.log().sum(dim=-1)

    def _eder_subsampling(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Subsample data using EDER method (not implemented)."""
        
        total_size = len(data['next_observation'])
        
        
        if self.n_subsamples >= total_size:
            utils.ColorPrint.yellow(f"Requested subsample size {self.n_subsamples} >= total size {total_size}. Returning full data.")
            return data
        
        utils.ColorPrint.green(f"Subsampling {self.n_subsamples} from {total_size} transitions.")
        
        starting_search_time = time()

        tmp_max = -np.inf

        for i in range(100):
            # Random subsampling (can be enhanced with trajectory-aware sampling)
            indices = np.random.choice(total_size, size=self.n_subsamples, replace=False)
            
            sampled_data = {
                'observation': data['observation'][indices],
                'action': data['action'][indices],
                'next_observation': data['next_observation'][indices],
                'proprio_observation': data['proprio_observation'][indices] if 'proprio_observation' in data else torch.empty((self.n_subsamples, 0), device=self.device, dtype=self.data_type),
                'alpha': data['alpha'][indices]
            }

            # Encoding state-action pairs as unique integers
            action_onehot = F.one_hot(sampled_data['action'].long(), self.n_actions).reshape(-1, self.n_actions)  # [B, |A|]
            
            # Outer product: [B, d] ⊗ [B, |A|] -> [B, d*|A|]
            encoded_sa = torch.einsum('bd,ba->bda', sampled_data['observation'], action_onehot).reshape(self.n_subsamples, -1)

            kernel_sa = encoded_sa@encoded_sa.T # [B, B]

            log_det =self.spd_logdet_cholesky(kernel_sa)

            # if np.random.rand() <= np.exp(log_det - self.max_log_det):
            # if log_det > self.max_log_det:

            #     if self.max_log_det == -np.inf:
            #         ColorPrint.green(f"EDER subsampling with log-det: {log_det.item():.4f}  in {time() - starting_search_time:.2f}s accepted after {i+1} attempts")
            #     else:
            #         ColorPrint.green(f"EDER subsampling with log-det: {log_det.item():.4f} (max: {self.max_log_det:.4f}) in {time() - starting_search_time:.2f}s accepted after {i+1} attempts")
            #     if log_det > self.max_log_det:
            #         self.max_log_det = log_det.item()
            #     return sampled_data
            
            if log_det > tmp_max:
                tmp_max = log_det.item()
                best_sampled_data = sampled_data


            if log_det > self.max_log_det:
                self.max_log_det = log_det.item()
        ColorPrint.red(f"EDER subsampling failed to find better subset; returning best sampled subset, with log-det: {tmp_max:.4f} (max: {self.max_log_det:.4f}) in {time() - starting_search_time:.2f}s")
        return best_sampled_data
    
    def _plot_horizon_histogram(self):
        """Plot histogram of dynamic horizons and save to file."""
        if not self.dynamic_horizon or len(self._horizon_history) == 0:
            return
        
        save_dir = os.path.join(os.getcwd(), "horizon_plots")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"horizon_histogram_period_{self._plot_counter}.png")
        
        # Get range of horizons
        min_horizon = min(self._horizon_history)
        max_horizon = max(self._horizon_history)
        
        # Create bins that include all integer values from min to max
        bins = np.arange(min_horizon - 0.5, max_horizon + 1.5, 1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(self._horizon_history, bins=bins, edgecolor='black', alpha=0.7, align='mid')
        
        # Set x-axis ticks to show each integer horizon value
        plt.xticks(range(min_horizon, max_horizon + 1))
        
        plt.xlabel('Target Horizon')
        plt.ylabel('Frequency')
        plt.title(f'Dynamic Horizon Distribution - Period {self._plot_counter}\n'
                  f'Total trajectories: {len(self._horizon_history)}, '
                  f'Mean: {np.mean(self._horizon_history):.2f}, '
                  f'Std: {np.std(self._horizon_history):.2f}')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        ColorPrint.green(f"Saved horizon histogram to {save_path}")
        
        # Increment counter for next plot
        self._plot_counter += 1

#************ KERNELS ************#
import jax
import jax.numpy as jnp


def pairwise_squared_distance_torch(X, Y):
    X_norm = torch.sum(X * X, dim=1, keepdim=True)
    Y_norm = torch.sum(Y * Y, dim=1, keepdim=True).T
    dist = X_norm + Y_norm - 2.0 * (X @ Y.T)
    return torch.clamp(dist, min=0.0)


def inner_product_kernel_torch(X, Y, bandwidth=None, distance_norm=None):
    del bandwidth
    del distance_norm
    return X @ Y.T


def gaussian_kernel_torch(X, Y, bandwidth=1.0, distance_norm="l2"):
    if distance_norm == "l1":
        distance = torch.cdist(X, Y, p=1)
        squared_distance = distance * distance
    else:
        squared_distance = pairwise_squared_distance_torch(X, Y)
    bandwidth = max(float(bandwidth), 1e-12)
    return torch.exp(-squared_distance / (2.0 * bandwidth * bandwidth))


def laplacian_kernel_torch(X, Y, bandwidth=1.0, distance_norm=None):
    del distance_norm
    distance = torch.cdist(X, Y, p=1)
    return torch.exp(-distance / max(float(bandwidth), 1e-12))


def dirac_kernel_torch(X, Y, bandwidth=None, distance_norm=None):
    del bandwidth
    del distance_norm
    return torch.all(X[:, None, :] == Y[None, :, :], dim=-1).to(dtype=X.dtype)


class KernelFunction:
    def __init__(
        self,
        kernel_type="inner_product",
        bandwidth=None,
        bandwidth_percentile=None,
        bandwidth_fit_norm="l2",
    ):
        kernel_type = str(kernel_type or "inner_product").strip().lower()
        aliases = {
            "inner": "inner_product",
            "linear": "inner_product",
            "dot": "inner_product",
            "dot_product": "inner_product",
            "rbf": "gaussian",
            "gaussian_kernel": "gaussian",
            "abel": "laplacian",
            "abel_diag": "laplacian",
            "laplace": "laplacian",
        }
        kernel_type = aliases.get(kernel_type, kernel_type)
        kernels = {
            "inner_product": inner_product_kernel_torch,
            "gaussian": gaussian_kernel_torch,
            "laplacian": laplacian_kernel_torch,
            "dirac": dirac_kernel_torch,
        }
        if kernel_type not in kernels:
            choices = ", ".join(sorted(kernels))
            raise ValueError(f"Unknown kernel_type={kernel_type!r}. Available choices: {choices}")

        self.kernel_type = kernel_type
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.bandwidth_percentile = None if bandwidth_percentile is None else float(bandwidth_percentile)
        self.bandwidth_fit_norm = str(bandwidth_fit_norm or "l2").strip().lower()
        if self.bandwidth_fit_norm not in ("l1", "l2"):
            raise ValueError("bandwidth_fit_norm must be 'l1' or 'l2'")
        self.bandwidth_fit_max_pairs = 50_000
        self._kernel = kernels[kernel_type]

    def __call__(self, X, Y):
        self.fit_bandwidth(X, Y)
        bandwidth = 1.0 if self.bandwidth is None else self.bandwidth
        return self._kernel(X, Y, bandwidth=bandwidth, distance_norm=self.bandwidth_fit_norm)

    def reset_auto_bandwidth(self):
        if self.bandwidth_percentile is not None:
            self.bandwidth = None

    def fit_bandwidth(self, X, Y):
        if self.kernel_type == "gaussian":
            self._maybe_fit_gaussian_bandwidth(X, Y)
        return self.bandwidth

    def _maybe_fit_gaussian_bandwidth(self, X, Y):
        if self.bandwidth is not None or self.bandwidth_percentile is None:
            return
        X_detached = X.detach()
        Y_detached = Y.detach()
        total_pairs = int(X_detached.shape[0]) * int(Y_detached.shape[0])
        if total_pairs <= self.bandwidth_fit_max_pairs:
            if self.bandwidth_fit_norm == "l1":
                distances = torch.cdist(X_detached, Y_detached, p=1)
            else:
                distances = torch.sqrt(torch.clamp(pairwise_squared_distance_torch(X_detached, Y_detached), min=0.0))
            fit_source = f"all {total_pairs} pairwise distances"
        else:
            sample_size = min(self.bandwidth_fit_max_pairs, total_pairs)
            x_idx = torch.randint(X_detached.shape[0], (sample_size,), device=X_detached.device)
            y_idx = torch.randint(Y_detached.shape[0], (sample_size,), device=Y_detached.device)
            ord_value = 1 if self.bandwidth_fit_norm == "l1" else 2
            distances = torch.linalg.vector_norm(X_detached[x_idx] - Y_detached[y_idx], ord=ord_value, dim=1)
            fit_source = f"{sample_size} sampled distances from {total_pairs} pairs"
        distances = distances[distances > 0]
        if distances.numel() == 0:
            self.bandwidth = 1.0
            return
        percentile = min(max(self.bandwidth_percentile / 100.0, 0.0), 1.0)
        self.bandwidth = float(torch.quantile(distances.flatten(), percentile).item())
        print(
            f"Fitted Gaussian kernel bandwidth={self.bandwidth:.6g} from "
            f"percentile={self.bandwidth_percentile} using {fit_source} "
            f"with {self.bandwidth_fit_norm} norm."
        )


def build_kernel_fn(
    kernel_type="inner_product",
    bandwidth=None,
    bandwidth_percentile=None,
    bandwidth_fit_norm="l2",
):
    return KernelFunction(
        kernel_type=kernel_type,
        bandwidth=bandwidth,
        bandwidth_percentile=bandwidth_percentile,
        bandwidth_fit_norm=bandwidth_fit_norm,
    )


# compute the dirac kernel on batches of states
@jax.jit    
def dirac_kernel(X, Y):
    return ((X.reshape(-1, 1) - Y.reshape(1, -1)) == 0) * 1.0


# gaussian kernel for matrices of n points and d dimensions
class gaussian_kernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        return jnp.exp(
            -(1 / self.sigma)
            * jnp.linalg.norm(
                X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1), axis=2
            )
        )


# gaussian kernel for matrices of n points and d dimension with a different sigma for each dimension
class gaussian_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
        return jnp.exp(
            -jnp.sum(
                (X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)) ** 2
                / (2 * self.sigma**2),
                axis=2,
            )
        )


class abel_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
        return jnp.exp(
            -jnp.sum(
                jnp.abs(X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1))
                / (jnp.sqrt(2) * self.sigma),
                axis=2,
            )
        )
    
class softmax:

    def __init__(self):
        pass

    def __call__(self, x):
        return jax.nn.softmax(x, axis=1)
