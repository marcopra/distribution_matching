from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from sampling import sample_time_steps


class EncodedTransitionFIFO:
    """FIFO storage for actor-ready encoded transitions.

    The first transition is pinned separately so actor batches can always place
    it at index 0 without duplicating it in the random sample.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("encoded FIFO capacity must be positive")
        self.capacity = int(capacity)
        self._data = None
        self._ids = None
        self._first = None
        self._first_id = None
        self._trajectory_ids = None
        self._trajectory_steps = None
        self._first_trajectory_id = None
        self._first_trajectory_step = None
        self._next_trajectory_id = 0
        self._next_trajectory_step = 0
        self.last_sampled_time_steps = None
        self.last_sampled_horizons = None
        self.last_sampled_trajectory_ids = None
        self.last_pivoted_cholesky_residuals = None
        self.last_pivoted_cholesky_candidate_count = None
        self.last_pivoted_cholesky_bandwidth = None

    def __len__(self):
        size = 0 if self._ids is None else int(self._ids.numel())
        return size + (1 if self._first is not None else 0)

    @property
    def has_first(self):
        return self._first is not None

    @property
    def data_count(self):
        return 0 if self._ids is None else int(self._ids.numel())

    @staticmethod
    def _index(encoded, index):
        return {key: value[index] for key, value in encoded.items()}

    @staticmethod
    def _cat(encoded_batches):
        keys = encoded_batches[0].keys()
        return {
            key: torch.cat([batch[key] for batch in encoded_batches], dim=0)
            for key in keys
        }

    def _trajectory_metadata(self, size, terminal_mask=None):
        terminal = np.zeros(size, dtype=bool) if terminal_mask is None else np.asarray(
            terminal_mask, dtype=bool
        ).reshape(-1)
        if terminal.shape[0] != size:
            raise ValueError("terminal_mask must align with transition_ids")

        previous_terminal = np.concatenate(([False], terminal[:-1]))
        trajectory_offsets = np.cumsum(previous_terminal, dtype=np.int64)
        trajectory_ids = self._next_trajectory_id + trajectory_offsets

        positions = np.arange(size, dtype=np.int64)
        segment_starts = np.where(previous_terminal, positions, -self._next_trajectory_step)
        segment_starts = np.maximum.accumulate(segment_starts)
        trajectory_steps = positions - segment_starts

        self._next_trajectory_id += int(terminal.sum())
        self._next_trajectory_step = 0 if size and terminal[-1] else int(trajectory_steps[-1] + 1)
        return torch.from_numpy(trajectory_ids), torch.from_numpy(trajectory_steps)

    def add(self, transition_ids, encoded, terminal_mask=None):
        transition_ids = torch.as_tensor(transition_ids, dtype=torch.long, device='cpu')
        trajectory_ids, trajectory_steps = self._trajectory_metadata(
            int(transition_ids.numel()), terminal_mask
        )
        encoded = {
            key: value.detach().to('cpu')
            for key, value in encoded.items()
        }

        first_mask = transition_ids == 0
        if first_mask.any():
            first_idx = int(torch.nonzero(first_mask, as_tuple=False)[0].item())
            self._first = self._index(encoded, slice(first_idx, first_idx + 1))
            self._first_id = int(transition_ids[first_idx].item())
            self._first_trajectory_id = int(trajectory_ids[first_idx].item())
            self._first_trajectory_step = int(trajectory_steps[first_idx].item())

        if self._first_id is None:
            keep_mask = torch.ones_like(transition_ids, dtype=torch.bool)
        else:
            keep_mask = transition_ids != self._first_id
        if not keep_mask.any():
            return

        new_ids = transition_ids[keep_mask]
        new_trajectory_ids = trajectory_ids[keep_mask]
        new_trajectory_steps = trajectory_steps[keep_mask]
        new_data = self._index(encoded, keep_mask)
        if self._data is None:
            self._ids = new_ids
            self._trajectory_ids = new_trajectory_ids
            self._trajectory_steps = new_trajectory_steps
            self._data = new_data
        else:
            self._ids = torch.cat([self._ids, new_ids], dim=0)
            self._trajectory_ids = torch.cat([self._trajectory_ids, new_trajectory_ids], dim=0)
            self._trajectory_steps = torch.cat([self._trajectory_steps, new_trajectory_steps], dim=0)
            self._data = self._cat([self._data, new_data])

        overflow = int(self._ids.numel()) - max(0, self.capacity - (1 if self._first is not None else 0))
        if overflow > 0:
            self._ids = self._ids[overflow:]
            self._trajectory_ids = self._trajectory_ids[overflow:]
            self._trajectory_steps = self._trajectory_steps[overflow:]
            self._data = self._index(self._data, slice(overflow, None))

    def _all_with_trajectory_metadata(self):
        batches = []
        trajectory_ids = []
        trajectory_steps = []
        if self._first is not None:
            batches.append(self._first)
            trajectory_ids.append(torch.tensor([self._first_trajectory_id], dtype=torch.long))
            trajectory_steps.append(torch.tensor([self._first_trajectory_step], dtype=torch.long))
        if self._data is not None and self.data_count > 0:
            batches.append(self._data)
            trajectory_ids.append(self._trajectory_ids)
            trajectory_steps.append(self._trajectory_steps)
        if not batches:
            raise RuntimeError("Encoded actor FIFO is empty")
        encoded = batches[0] if len(batches) == 1 else self._cat(batches)
        return encoded, torch.cat(trajectory_ids), torch.cat(trajectory_steps)

    @staticmethod
    def _estimate_gaussian_bandwidth(features, multiplier, max_points=1000):
        """Estimate median-distance bandwidth from a bounded candidate subset."""
        features = features.detach().to(device="cpu", dtype=torch.float32).reshape(features.shape[0], -1)
        if features.shape[0] > max_points:
            features = features[torch.randperm(features.shape[0])[:max_points]]
        distances = torch.pdist(features, p=2)
        distances = distances[distances > 0]
        median = 1.0 if distances.numel() == 0 else float(torch.median(distances).item())
        return max(median * float(multiplier), 1e-12)

    @staticmethod
    def _pivoted_cholesky_indices(
        features,
        size,
        tolerance,
        force_first,
        kernel_type="inner_product",
        actions=None,
        bandwidth=None,
        show_progress=True,
    ):
        """Select diverse rows for an inner-product or Gaussian Nyström approximation."""
        features = features.detach().to(device="cpu", dtype=torch.float32).reshape(features.shape[0], -1)
        kernel_type = str(kernel_type).lower()
        if kernel_type not in ("inner_product", "gaussian"):
            raise ValueError("pivoted Cholesky supports inner_product and gaussian kernels")
        if actions is not None:
            actions = torch.as_tensor(actions, dtype=torch.long, device="cpu").reshape(-1)
            if actions.shape[0] != features.shape[0]:
                raise ValueError("actions must align with pivoted Cholesky features")
        if kernel_type == "gaussian":
            if bandwidth is None or bandwidth <= 0.0:
                raise ValueError("Gaussian pivoted Cholesky requires a positive bandwidth")
            if actions is None:
                raise ValueError("Gaussian pivoted Cholesky requires action indices")
        count = int(features.shape[0])
        target = min(int(size), count)
        if target <= 0:
            return torch.empty(0, dtype=torch.long), np.empty(0, dtype=np.float32)

        residual = (
            torch.sum(features * features, dim=1)
            if kernel_type == "inner_product"
            else torch.ones(count, dtype=features.dtype)
        )
        scale = max(float(residual.max().item()), torch.finfo(residual.dtype).eps)
        stop_threshold = float(tolerance) * scale
        factor = torch.zeros((count, target), dtype=features.dtype)
        available = torch.ones(count, dtype=torch.bool)
        selected = []
        pivot_residuals = []

        with tqdm(
            total=target,
            desc=f"Pivoted Cholesky ({kernel_type})",
            unit="pivot",
            leave=False,
            disable=not bool(show_progress),
        ) as progress:
            progress.set_postfix(candidates=count, threshold=f"{stop_threshold:.2e}")
            for column_index in range(target):
                if column_index == 0 and force_first:
                    pivot = 0
                else:
                    scores = residual.masked_fill(~available, -torch.inf)
                    pivot = int(torch.argmax(scores).item())
                    if float(scores[pivot].item()) <= stop_threshold:
                        progress.set_postfix(
                            candidates=count,
                            residual=f"{float(scores[pivot].item()):.2e}",
                            stopped="tolerance",
                        )
                        break

                pivot_residual = max(float(residual[pivot].item()), 0.0)
                selected.append(pivot)
                pivot_residuals.append(pivot_residual)
                available[pivot] = False
                progress.update(1)
                progress.set_postfix(
                    candidates=count,
                    residual=f"{pivot_residual:.2e}",
                    threshold=f"{stop_threshold:.2e}",
                )

                # A forced initial point can have zero kernel norm. Keep it in row
                # zero for alpha support, but do not use it as a Cholesky direction.
                if pivot_residual <= torch.finfo(features.dtype).eps:
                    residual[pivot] = 0.0
                    continue

                if kernel_type == "inner_product":
                    kernel_column = features @ features[pivot]
                    # For compact state-action data, ψ(s,a) is represented by
                    # φ(s) plus an action id. Its inner product is zero across
                    # different actions.
                    if actions is not None:
                        kernel_column *= actions == actions[pivot]
                else:
                    squared_distance = torch.sum(
                        (features - features[pivot]).square(), dim=1
                    )
                    kernel_column = torch.exp(
                        -squared_distance / (2.0 * float(bandwidth) ** 2)
                    )
                    kernel_column *= actions == actions[pivot]
                if column_index:
                    kernel_column -= factor[:, :column_index] @ factor[pivot, :column_index]
                factor[:, column_index] = kernel_column / np.sqrt(pivot_residual)
                residual.sub_(factor[:, column_index].square()).clamp_(min=0.0)
                residual[pivot] = 0.0

        return (
            torch.as_tensor(selected, dtype=torch.long),
            np.asarray(pivot_residuals, dtype=np.float32),
        )

    def _sample_pivoted_cholesky(
        self,
        size,
        device,
        include_first,
        candidate_multiplier,
        tolerance,
        kernel_type,
        kernel_bandwidth,
        kernel_bandwidth_mult,
        show_progress,
    ):
        if size <= 0:
            raise ValueError("sample size must be positive")
        if candidate_multiplier < 1.0:
            raise ValueError("candidate_multiplier must be at least 1")
        if tolerance < 0.0:
            raise ValueError("pivoted Cholesky tolerance must be non-negative")

        encoded, _, _ = self._all_with_trajectory_metadata()
        total = int(encoded["phi_obs"].shape[0])
        target = min(int(size), total)
        force_first = bool(include_first and self._first is not None)
        candidate_count = min(
            total,
            max(target, int(np.ceil(float(candidate_multiplier) * target))),
        )

        if force_first:
            remaining = candidate_count - 1
            candidate_indices = torch.cat([
                torch.zeros(1, dtype=torch.long),
                torch.randperm(max(0, total - 1))[:remaining] + 1,
            ])
        else:
            candidate_indices = torch.randperm(total)[:candidate_count]

        candidates = self._index(encoded, candidate_indices)
        kernel_type = str(kernel_type).lower()
        if kernel_type == "inner_product":
            if "psi" in candidates:
                features = candidates["psi"]
                actions = None
            elif "action" in candidates:
                features = candidates["phi_obs"]
                actions = candidates["action"].reshape(-1)
            else:
                raise KeyError("encoded FIFO requires either psi or compact action data")
            bandwidth = None
        elif kernel_type == "gaussian":
            features = candidates["phi_obs"]
            actions = (
                candidates["action"].reshape(-1)
                if "action" in candidates
                else torch.argmax(candidates["E"], dim=1)
            )
            if kernel_bandwidth is not None:
                bandwidth = float(kernel_bandwidth)
                if bandwidth <= 0.0:
                    raise ValueError("kernel_bandwidth must be positive when set")
            else:
                multiplier = 1.0 if kernel_bandwidth_mult is None else float(kernel_bandwidth_mult)
                if multiplier <= 0.0:
                    raise ValueError("kernel_bandwidth_mult must be positive when set")
                bandwidth = self._estimate_gaussian_bandwidth(features, multiplier)
        else:
            raise ValueError(
                "pivoted Cholesky supports kernel_type=inner_product or gaussian"
            )
        local_indices, residuals = self._pivoted_cholesky_indices(
            features,
            target,
            tolerance,
            force_first=force_first,
            kernel_type=kernel_type,
            actions=actions,
            bandwidth=bandwidth,
            show_progress=show_progress,
        )
        selected = candidate_indices[local_indices]
        self.last_pivoted_cholesky_residuals = residuals
        self.last_pivoted_cholesky_candidate_count = candidate_count
        self.last_pivoted_cholesky_bandwidth = bandwidth
        sampled = self._index(encoded, selected)
        return {key: value.to(device) for key, value in sampled.items()}

    def sample_by_strategy(
        self,
        size,
        device,
        strategy="random",
        gamma=0.99,
        include_first=True,
        candidate_multiplier=5.0,
        cholesky_tolerance=1e-6,
        kernel_type="inner_product",
        kernel_bandwidth=None,
        kernel_bandwidth_mult=None,
        cholesky_progress=True,
    ):
        strategy = str(strategy).lower()
        self.last_pivoted_cholesky_residuals = None
        self.last_pivoted_cholesky_candidate_count = None
        self.last_pivoted_cholesky_bandwidth = None
        if strategy == "random":
            self.last_sampled_time_steps = None
            self.last_sampled_horizons = None
            self.last_sampled_trajectory_ids = None
            return self.sample(size, device, include_first=include_first)
        if strategy == "pivoted_cholesky":
            self.last_sampled_time_steps = None
            self.last_sampled_horizons = None
            self.last_sampled_trajectory_ids = None
            return self._sample_pivoted_cholesky(
                size,
                device,
                include_first,
                candidate_multiplier,
                cholesky_tolerance,
                kernel_type,
                kernel_bandwidth,
                kernel_bandwidth_mult,
                cholesky_progress,
            )
        if strategy not in ("gamma_h", "reverse_gamma_h"):
            raise ValueError(
                "strategy must be one of: random, gamma_h, reverse_gamma_h, pivoted_cholesky"
            )
        if size <= 0:
            raise ValueError("sample size must be positive")

        encoded, trajectory_ids, _ = self._all_with_trajectory_metadata()
        total = int(trajectory_ids.numel())
        remaining = int(size) - (1 if include_first and self._first is not None else 0)
        if remaining < 0:
            remaining = 0

        boundary = torch.ones(total, dtype=torch.bool)
        boundary[1:] = trajectory_ids[1:] != trajectory_ids[:-1]
        starts = torch.nonzero(boundary, as_tuple=False).reshape(-1)
        ends = torch.cat([starts[1:], torch.tensor([total], dtype=torch.long)])
        horizons = ends - starts - 1

        chosen_groups = np.random.randint(0, int(starts.numel()), size=remaining)
        chosen_groups_t = torch.as_tensor(chosen_groups, dtype=torch.long)
        chosen_horizons = horizons[chosen_groups_t].numpy()
        sampled_t = sample_time_steps(
            gamma,
            remaining,
            horizon=chosen_horizons,
            rng=np.random,
        ) if remaining else np.empty(0, dtype=np.int64)
        selected_t = chosen_horizons - sampled_t if strategy == "reverse_gamma_h" else sampled_t
        selected = starts[chosen_groups_t] + torch.as_tensor(selected_t, dtype=torch.long)

        if include_first and self._first is not None:
            selected = torch.cat([torch.zeros(1, dtype=torch.long), selected])
            sampled_times = np.concatenate(([self._first_trajectory_step], selected_t))
            sampled_horizons = np.concatenate(([0], chosen_horizons))
            sampled_trajectories = np.concatenate((
                [self._first_trajectory_id],
                trajectory_ids[starts[chosen_groups_t]].numpy(),
            ))
        else:
            sampled_times = selected_t
            sampled_horizons = chosen_horizons
            sampled_trajectories = trajectory_ids[starts[chosen_groups_t]].numpy()

        self.last_sampled_time_steps = np.asarray(sampled_times, dtype=np.int64)
        self.last_sampled_horizons = np.asarray(sampled_horizons, dtype=np.int64)
        self.last_sampled_trajectory_ids = np.asarray(sampled_trajectories, dtype=np.int64)
        sampled = self._index(encoded, selected)
        return {key: value.to(device) for key, value in sampled.items()}

    def sample(self, size, device, include_first=True):
        if size <= 0:
            raise ValueError("sample size must be positive")
        if len(self) == 0:
            raise RuntimeError("Encoded actor FIFO is empty")

        batches = []
        remaining = int(size)
        if include_first and self._first is not None:
            batches.append(self._first)
            remaining -= 1

        data_size = 0 if self._ids is None else int(self._ids.numel())
        if remaining > 0 and data_size > 0:
            take = min(remaining, data_size)
            indices = torch.randperm(data_size)[:take]
            batches.append(self._index(self._data, indices))

        if not batches:
            raise RuntimeError("Encoded actor FIFO does not contain a first transition yet")

        sampled = batches[0] if len(batches) == 1 else self._cat(batches)
        return {
            key: value.to(device)
            for key, value in sampled.items()
        }

    def all(self, device, include_first=True):
        if len(self) == 0:
            raise RuntimeError("Encoded actor FIFO is empty")

        batches = []
        if include_first and self._first is not None:
            batches.append(self._first)
        if self._data is not None and self.data_count > 0:
            batches.append(self._data)

        if not batches:
            raise RuntimeError("Encoded actor FIFO does not contain a first transition yet")

        encoded = batches[0] if len(batches) == 1 else self._cat(batches)
        return {
            key: value.to(device)
            for key, value in encoded.items()
        }
