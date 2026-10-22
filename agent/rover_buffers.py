from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from sampling import sample_time_steps


class EncodedTransitionFIFO:
    """FIFO storage for actor-ready encoded transitions.

    The first transition is pinned separately so actor batches can always place
    it at index 0 without duplicating it in a sample.
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

    def _move_storage_to_cpu(self):
        """Keep FIFO storage on CPU, including tensors restored via CUDA map_location."""
        if self._ids is not None:
            self._ids = self._ids.cpu()
        if self._trajectory_ids is not None:
            self._trajectory_ids = self._trajectory_ids.cpu()
        if self._trajectory_steps is not None:
            self._trajectory_steps = self._trajectory_steps.cpu()
        if self._data is not None:
            self._data = {key: value.cpu() for key, value in self._data.items()}
        if self._first is not None:
            self._first = {key: value.cpu() for key, value in self._first.items()}

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
        # Checkpoints load through map_location=self.device, which can place
        # these FIFO tensors on CUDA. New transition IDs and encoded rows are
        # deliberately CPU-backed, so restore the FIFO's storage invariant.
        self._move_storage_to_cpu()
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
            self._first = {key: value[first_idx:first_idx + 1].clone() for key, value in encoded.items()}
            self._first_id = int(transition_ids[first_idx].item())
            self._first_trajectory_id = int(trajectory_ids[first_idx].item())
            self._first_trajectory_step = int(trajectory_steps[first_idx].item())

        if self._first_id is None:
            keep_mask = torch.ones_like(transition_ids, dtype=torch.bool)
        else:
            keep_mask = transition_ids != self._first_id
        capacity = self.capacity - int(self.has_first)
        keep = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)[-capacity:] if capacity else torch.empty(0, dtype=torch.long)
        old_count = min(self.data_count, capacity - len(keep))
        old_rows = slice(self.data_count - old_count, None)
        def append(old, new):
            return torch.cat((old[old_rows], new[keep]), dim=0)
        assert all(value.shape[0] == len(transition_ids) for value in encoded.values())
        if self._data is not None:
            assert self._data.keys() == encoded.keys(), "FIFO columns changed"
            assert all(value.shape[1:] == self._data[key].shape[1:] and value.dtype == self._data[key].dtype
                       for key, value in encoded.items())
        if self._data is None:
            self._ids = transition_ids[keep]
            self._trajectory_ids = trajectory_ids[keep]
            self._trajectory_steps = trajectory_steps[keep]
            self._data = self._index(encoded, keep)
        else:
            self._ids = append(self._ids, transition_ids)
            self._trajectory_ids = append(self._trajectory_ids, trajectory_ids)
            self._trajectory_steps = append(self._trajectory_steps, trajectory_steps)
            for key, value in encoded.items():
                self._data[key] = append(self._data[key], value)

    def _all_with_trajectory_metadata(self, keys=None):
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
        if keys is not None:
            batches = [{key: value for key, value in batch.items() if key in keys} for batch in batches]
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
    @torch.no_grad()
    def _pivoted_cholesky_indices(
        features,
        size,
        tolerance,
        force_first,
        kernel_type="inner_product",
        actions=None,
        bandwidth=None,
        show_progress=False,
        device=None,
    ):
        """Greedy global pivots with contiguous, action-local Cholesky factors."""
        assert features.ndim >= 2 and features.is_floating_point()
        assert int(size) >= 0 and np.isfinite(tolerance) and tolerance >= 0
        features = features.detach().to(device=device, dtype=torch.float32).reshape(features.shape[0], -1)
        assert torch.isfinite(features).all(), "Non-finite Cholesky features"
        kernel_type = str(kernel_type).lower()
        if kernel_type not in ("inner_product", "gaussian"):
            raise ValueError("pivoted Cholesky supports inner_product and gaussian kernels")
        if actions is not None:
            actions = torch.as_tensor(actions, device="cpu").reshape(-1)
            assert not actions.is_floating_point() and not actions.is_complex()
            assert (actions >= 0).all(), "Negative Cholesky action index"
            if actions.shape[0] != features.shape[0]:
                raise ValueError("actions must align with pivoted Cholesky features")
        if kernel_type == "gaussian":
            if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0.0:
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
            else features.new_ones(count)
        )
        scale = max(float(residual.max().item()), torch.finfo(residual.dtype).eps)
        stop_threshold = float(tolerance) * scale
        assert torch.isfinite(residual).all(), "Non-finite Cholesky diagonal"
        # Preserve original candidate order inside each action, including ties.
        rows = ([torch.arange(count)] if actions is None else
                [torch.nonzero(actions == action, as_tuple=True)[0] for action in torch.unique(actions)])
        routing = torch.empty((count, 2), dtype=torch.long)
        for group, indices in enumerate(rows):
            routing[indices, 0] = group
            routing[indices, 1] = torch.arange(len(indices))
        rows = [indices.to(features.device) for indices in rows]
        blocks = [features.index_select(0, indices) for indices in rows]
        # Grow only the selected action's factor. Unwritten directions are never read.
        factors = [features.new_empty((0, len(indices))) for indices in rows]
        ranks = [0] * len(rows)
        available = torch.ones(count, dtype=torch.bool, device=features.device)
        del features
        selected = []
        pivot_residuals = []

        with tqdm(
            total=target,
            desc=f"Pivoted Cholesky ({kernel_type})",
            unit="pivot",
            leave=False,
            disable=not bool(show_progress),
        ) as progress:
            progress.set_postfix(candidates=count, threshold=f"{stop_threshold:.2e}", refresh=False)
            for column_index in range(target):
                if column_index == 0 and force_first:
                    pivot = 0
                    pivot_residual = float(residual[pivot].item())
                else:
                    value, index = residual.masked_fill(~available, -torch.inf).max(dim=0)
                    # One device-to-host synchronization for both value and index.
                    pivot_residual, pivot = torch.stack((value.double(), index.double())).tolist()
                    pivot = int(pivot)
                    if tolerance > 0.0 and pivot_residual <= stop_threshold:
                        progress.set_postfix(
                            candidates=count,
                            residual=f"{pivot_residual:.2e}",
                            stopped="tolerance",
                            refresh=False,
                        )
                        break

                assert np.isfinite(pivot_residual), "Non-finite Cholesky residual"
                pivot_residual = max(pivot_residual, 0.0)
                selected.append(pivot)
                pivot_residuals.append(pivot_residual)
                available[pivot] = False
                if progress.update(1):
                    progress.set_postfix(
                        candidates=count,
                        residual=f"{pivot_residual:.2e}",
                        threshold=f"{stop_threshold:.2e}",
                        refresh=False,
                    )

                # A forced initial point can have zero kernel norm. Keep it in row
                # zero for alpha support, but do not use it as a Cholesky direction.
                if pivot_residual <= torch.finfo(residual.dtype).eps:
                    residual[pivot] = 0.0
                    continue

                group, local_pivot = routing[pivot].tolist()
                block, rank, factor = blocks[group], ranks[group], factors[group]
                if rank == len(factor):
                    capacity = min(target, len(block), max(32, 2 * rank))
                    grown = block.new_empty((capacity, len(block)))
                    grown[:rank].copy_(factor)
                    factors[group] = factor = grown
                if kernel_type == "inner_product":
                    kernel_column = block @ block[local_pivot]
                else:
                    kernel_column = (block - block[local_pivot]).square_().sum(dim=1)
                    kernel_column.neg_().div_(2.0 * float(bandwidth) ** 2).exp_()
                if rank:
                    kernel_column -= factor[:rank].T @ factor[:rank, local_pivot]
                factor[rank].copy_(kernel_column).div_(np.sqrt(pivot_residual))
                indices = rows[group]
                updated = residual.index_select(0, indices)
                updated.sub_(factor[rank].square()).clamp_(min=0.0)
                residual.index_copy_(0, indices, updated)
                ranks[group] += 1
                residual[pivot] = 0.0

        assert torch.isfinite(residual).all(), "Non-finite Cholesky residual"
        return (
            torch.as_tensor(selected, dtype=torch.long),
            np.asarray(pivot_residuals, dtype=np.float32),
        )

    def _pivoted_cholesky_sample_indices(
        self,
        encoded,
        size,
        force_first,
        candidate_multiplier,
        tolerance,
        kernel_type,
        kernel_bandwidth,
        kernel_bandwidth_mult,
        show_progress,
        kernel_feature_key=None,
        device=None,
    ):
        if size <= 0:
            raise ValueError("sample size must be positive")
        if candidate_multiplier < 1.0:
            raise ValueError("candidate_multiplier must be at least 1")
        if tolerance < 0.0:
            raise ValueError("pivoted Cholesky tolerance must be non-negative")

        total = int(encoded["phi_obs"].shape[0])
        target = min(int(size), total)
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

        feature_keys = {"phi_obs", "psi", "action", "E", kernel_feature_key}
        candidates = self._index({key: value for key, value in encoded.items() if key in feature_keys}, candidate_indices)
        kernel_type = str(kernel_type).lower()
        kernel_features = (
            candidates[kernel_feature_key]
            if kernel_feature_key is not None else None
        )
        if kernel_type == "inner_product":
            if kernel_features is not None:
                features = kernel_features
                actions = candidates.get("action")
                if actions is not None:
                    actions = actions.reshape(-1)
            elif "psi" in candidates:
                features = candidates["psi"]
                actions = None
            elif "action" in candidates:
                features = candidates["phi_obs"]
                actions = candidates["action"].reshape(-1)
            else:
                raise KeyError("encoded FIFO requires either psi or compact action data")
            bandwidth = None
        elif kernel_type == "gaussian":
            features = (
                candidates["phi_obs"]
                if kernel_features is None else kernel_features
            )
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
            device=device,
        )
        selected = candidate_indices[local_indices]
        self.last_pivoted_cholesky_residuals = residuals
        self.last_pivoted_cholesky_candidate_count = candidate_count
        self.last_pivoted_cholesky_bandwidth = bandwidth
        return selected

    def _gather(self, indices, device):
        """Gather selected records without concatenating the full history."""
        pinned = indices == 0 if self.has_first else torch.zeros_like(indices, dtype=torch.bool)
        source = self._first if self.has_first else self._data
        result = {}
        for key, example in source.items():
            output = example.new_empty((len(indices), *example.shape[1:]))
            if pinned.any():
                output[pinned] = self._first[key]
            if (~pinned).any():
                output[~pinned] = self._data[key][indices[~pinned] - int(self.has_first)]
            result[key] = output.to(device)
        return result

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
        cholesky_progress=False,
        kernel_feature_key=None,
        support=None,
        cholesky_device=None,
    ):
        strategy = str(strategy).lower()
        self.last_pivoted_cholesky_residuals = None
        self.last_pivoted_cholesky_candidate_count = None
        self.last_pivoted_cholesky_bandwidth = None
        assert size > 0
        if strategy == "random" and support is None:
            self.last_sampled_time_steps = None
            self.last_sampled_horizons = None
            self.last_sampled_trajectory_ids = None
            return self.sample(size, device, include_first=include_first)
        if support is None:
            keys = {"phi_obs", "psi", "action", "E", kernel_feature_key}
            encoded, trajectory_ids, trajectory_steps = self._all_with_trajectory_metadata(keys)
            has_first = self.has_first
        else:
            encoded = support
            trajectory_ids = torch.zeros(len(encoded["phi_obs"]), dtype=torch.long)
            trajectory_steps = torch.arange(len(encoded["phi_obs"]))
            has_first = True
        pin_first = bool(include_first and has_first)
        total = len(encoded["phi_obs"])
        if strategy in ("random", "pivoted_cholesky"):
            self.last_sampled_time_steps = None
            self.last_sampled_horizons = None
            self.last_sampled_trajectory_ids = None
            if strategy == "random":
                selected = torch.randperm(total - int(has_first))[:max(0, size - int(pin_first))] + int(has_first)
                if pin_first:
                    selected = torch.cat((torch.zeros(1, dtype=torch.long), selected))
            else:
                selected = self._pivoted_cholesky_sample_indices(
                    encoded, size, pin_first, candidate_multiplier, cholesky_tolerance,
                    kernel_type, kernel_bandwidth, kernel_bandwidth_mult,
                    cholesky_progress, kernel_feature_key,
                    device=device if cholesky_device is None else cholesky_device,
                )
            return ({key: value[selected].to(device) for key, value in support.items()}
                    if support is not None else self._gather(selected, device))
        assert strategy in ("gamma_h", "reverse_gamma_h"), f"Unknown subsampling strategy: {strategy}"
        remaining = max(0, int(size) - int(pin_first))
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

        if pin_first:
            selected = torch.cat([torch.zeros(1, dtype=torch.long), selected])
            sampled_times = np.concatenate(([int(trajectory_steps[0])], selected_t))
            sampled_horizons = np.concatenate(([0], chosen_horizons))
            sampled_trajectories = np.concatenate((
                [int(trajectory_ids[0])],
                trajectory_ids[starts[chosen_groups_t]].numpy(),
            ))
        else:
            sampled_times = selected_t
            sampled_horizons = chosen_horizons
            sampled_trajectories = trajectory_ids[starts[chosen_groups_t]].numpy()

        self.last_sampled_time_steps = np.asarray(sampled_times, dtype=np.int64)
        self.last_sampled_horizons = np.asarray(sampled_horizons, dtype=np.int64)
        self.last_sampled_trajectory_ids = np.asarray(sampled_trajectories, dtype=np.int64)
        return ({key: value[selected].to(device) for key, value in support.items()}
                if support is not None else self._gather(selected, device))

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

    def sample_recent(self, size, device):
        """Return the pinned first transition plus newest ``size`` transitions."""
        if size <= 0:
            raise ValueError("recent sample size must be positive")
        if self._first is None:
            raise RuntimeError("Encoded actor FIFO has no pinned first transition")

        batches = [self._first]
        if self.data_count:
            start = max(0, self.data_count - int(size))
            batches.append(self._index(self._data, slice(start, None)))

        sampled = batches[0] if len(batches) == 1 else self._cat(batches)
        return {key: value.to(device) for key, value in sampled.items()}

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
