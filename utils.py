import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def chain(*iterables):
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def pairwise_squared_distance(X, Y):
    X_norm = torch.sum(X * X, dim=1, keepdim=True)
    Y_norm = torch.sum(Y * Y, dim=1, keepdim=True).T
    dist = X_norm + Y_norm - 2.0 * (X @ Y.T)
    return torch.clamp(dist, min=0.0)


def inner_product_kernel(X, Y, bandwidth=None, distance_norm=None):
    del bandwidth
    del distance_norm
    return X @ Y.T


def gaussian_kernel_torch(X, Y, bandwidth=1.0, distance_norm=None):
    del distance_norm
    squared_distance = pairwise_squared_distance(X, Y)
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
            "inner_product": inner_product_kernel,
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
        self.bandwidth_fit_max_pairs = 16_000_000
        self._kernel = kernels[kernel_type]

    def __call__(self, X, Y):
        self.fit_bandwidth(X, Y)
        bandwidth = 1.0 if self.bandwidth is None else self.bandwidth
        return self._kernel(X, Y, bandwidth=bandwidth)

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
            distances = torch.sqrt(torch.clamp(pairwise_squared_distance(X_detached, Y_detached), min=0.0))
            fit_source = f"all {total_pairs} pairwise distances"
        else:
            sample_size = min(self.bandwidth_fit_max_pairs, total_pairs)
            x_idx = torch.randint(X_detached.shape[0], (sample_size,), device=X_detached.device)
            y_idx = torch.randint(Y_detached.shape[0], (sample_size,), device=Y_detached.device)
            distances = torch.linalg.vector_norm(X_detached[x_idx] - Y_detached[y_idx], ord=2, dim=1)
            fit_source = f"{sample_size} sampled distances from {total_pairs} pairs"
        distances = distances[distances > 0]
        if distances.numel() == 0:
            self.bandwidth = 1.0
            return
        percentile = min(max(self.bandwidth_percentile / 100.0, 0.0), 1.0)
        self.bandwidth = float(torch.quantile(distances.flatten(), percentile).item())
        print(
            f"Fitted Gaussian kernel bandwidth={self.bandwidth:.6g} from "
            f"percentile={self.bandwidth_percentile} using {fit_source} with l2 norm."
        )


def build_kernel_fn(
    kernel_type="inner_product",
    bandwidth=None,
    bandwidth_percentile=None,
):
    return KernelFunction(
        kernel_type=kernel_type,
        bandwidth=bandwidth,
        bandwidth_percentile=bandwidth_percentile,
    )


def grad_norm(params, norm_type=2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def param_norm(params, norm_type=2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(self.device)
            ) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(
                    self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward


class ColorPrint:
    @staticmethod
    def blue(text):
        print(f"\033[94m{text}\033[0m")
    
    @staticmethod
    def green(text):
        print(f"\033[92m{text}\033[0m")
    
    @staticmethod
    def yellow(text):
        print(f"\033[93m{text}\033[0m")
    
    @staticmethod
    def red(text):
        print(f"\033[91m{text}\033[0m")

def load_policy_weights_into_agent(agent, npy_path, device='cuda'):
    """
    Carica i pesi della policy da un file .npy nel layer lineare dell'actor.
    
    Args:
        agent: L'agente SAC con actor lineare
        npy_path: Path al file .npy contenente la matrice dei pesi (policy_operator)
        device: Device su cui caricare i pesi
    
    La matrice .npy deve avere shape (n_states * n_actions, n_states).
    Il layer lineare dell'actor avrà shape (n_actions, n_states).
    """
    
    # Carica la matrice dal file
    policy_operator = np.load(npy_path)
    print(f"Policy operator loaded from {npy_path}")
    print(f"Policy operator shape: {policy_operator.shape}")
    
    # Estrai n_states e n_actions dalla shape
    n_states_times_actions, n_states = policy_operator.shape
    n_actions = agent.action_dim
    
    # Verifica la compatibilità
    assert n_states_times_actions == n_states * n_actions, \
        f"Shape mismatch: expected {n_states * n_actions}, got {n_states_times_actions}"
    
    # Reshape la matrice in (n_states, n_actions, n_states)
    policy_3d = policy_operator.reshape((n_states, n_actions, n_states))
    
    # Estrai i pesi rilevanti per il layer lineare (elementi diagonali)
    # Per ogni stato s, prendiamo policy_3d[s, :, s] -> (n_actions,)
    policy_weights = np.zeros((n_actions, n_states))
    for s in range(n_states):
        policy_weights[:, s] = policy_3d[s, :, s]
    
    # Converti in tensor e carica nel layer lineare
    policy_weights_tensor = torch.FloatTensor(policy_weights).to(device)
    
    # Carica i pesi nel layer lineare dell'actor
    with torch.no_grad():
        agent.actor.policy.weight.copy_(policy_weights_tensor)
        # Opzionale: imposta il bias a zero se presente
        if agent.actor.policy.bias is not None:
            agent.actor.policy.bias.zero_()
    
    print(f"✓ Policy weights loaded successfully!")
    print(f"  Linear layer shape: {agent.actor.policy.weight.shape}")
    print(f"  Weights sum per state (should be ~1.0):")
    
    # Verifica che i pesi formino una distribuzione valida
    weights_sum = policy_weights_tensor.sum(dim=0).cpu().numpy()
    print(f"    Mean: {weights_sum.mean():.6f}, Std: {weights_sum.std():.6f}")
    print(f"    Min: {weights_sum.min():.6f}, Max: {weights_sum.max():.6f}")
    
    return agent
