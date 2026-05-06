from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

import gym_env


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        return {k: episode[k] for k in episode.keys()}


def decode_state_indices(observations):
    observations = np.asarray(observations)
    if observations.ndim != 2:
        raise ValueError(
            f"Expected one-hot observations with shape [T, n_states], got {observations.shape}"
        )
    return np.argmax(observations, axis=1)


def observation_key(observation):
    observation = np.ascontiguousarray(observation)
    return (observation.shape, observation.dtype.str, observation.tobytes())


def get_max_samples(cfg):
    first_n_elements = cfg.get("first_n_elements", None)
    max_samples = cfg.get("max_samples", None)

    if first_n_elements is not None and max_samples is not None:
        if int(first_n_elements) != int(max_samples):
            raise ValueError(
                "Set only one replay-buffer limit: first_n_elements or max_samples "
                f"(got first_n_elements={first_n_elements}, max_samples={max_samples})"
            )

    if first_n_elements is not None:
        max_samples = first_n_elements

    if max_samples is None:
        return None

    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError(
            "first_n_elements/max_samples must be positive or null, "
            f"got {max_samples}"
        )
    return max_samples


def limited_observations(observations, total_observations, max_samples):
    if max_samples is None:
        return observations

    remaining = max_samples - total_observations
    if remaining <= 0:
        return observations[:0]
    return observations[:remaining]


def render_all_state_observations(env):
    rendered_observations = []
    key_to_state = {}
    duplicate_states = []

    for state_idx in range(env.unwrapped.n_states):
        time_step = env.reset(options={"start_state": state_idx})
        observation = np.asarray(time_step.observation)
        rendered_observations.append(observation)

        key = observation_key(observation)
        if key in key_to_state:
            duplicate_states.append((key_to_state[key], state_idx))
        key_to_state[key] = state_idx

    if duplicate_states:
        pairs = ", ".join(f"{a}/{b}" for a, b in duplicate_states[:5])
        raise ValueError(
            "Pixel matching is ambiguous because some rendered states are identical: "
            f"{pairs}"
        )

    return np.stack(rendered_observations, axis=0), key_to_state


def nearest_rendered_state_indices(observations, rendered_observations, batch_size):
    observations = observations.reshape(observations.shape[0], -1).astype(np.float32)
    rendered = rendered_observations.reshape(rendered_observations.shape[0], -1).astype(np.float32)

    rendered_norms = np.sum(rendered * rendered, axis=1)
    nearest_indices = []
    nearest_distances = []
    batch_size = max(1, int(batch_size))

    for start in range(0, observations.shape[0], batch_size):
        batch = observations[start:start + batch_size]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        distances = batch_norms + rendered_norms[None, :] - 2.0 * batch @ rendered.T
        distances = np.maximum(distances, 0.0)
        indices = np.argmin(distances, axis=1)
        nearest_indices.append(indices)
        nearest_distances.append(distances[np.arange(len(indices)), indices])

    return np.concatenate(nearest_indices), np.concatenate(nearest_distances)


def count_one_hot_observations(npz_files, n_states, max_samples):
    state_counts = np.zeros(n_states, dtype=np.int64)
    total_observations = 0

    for npz_file in npz_files:
        episode = load_episode(npz_file)
        if "observation" not in episode:
            raise KeyError(f"Missing 'observation' in {npz_file}")

        observations = limited_observations(
            episode["observation"],
            total_observations,
            max_samples,
        )
        if len(observations) == 0:
            break

        state_indices = decode_state_indices(observations)
        if np.any(state_indices >= n_states):
            raise ValueError(
                f"Found decoded states outside the environment range in {npz_file}: "
                f"max index {state_indices.max()}, env has {n_states} states"
            )

        state_counts += np.bincount(state_indices, minlength=n_states)
        total_observations += len(state_indices)

    return state_counts, total_observations, {"mode": "one-hot"}


def count_pixel_observations(npz_files, env, max_samples, nearest_batch_size):
    rendered_observations, key_to_state = render_all_state_observations(env)
    rendered_shape = rendered_observations.shape[1:]
    n_states = rendered_observations.shape[0]
    state_counts = np.zeros(n_states, dtype=np.int64)
    total_observations = 0
    unmatched_counts = {}
    unmatched_observations = {}

    for npz_file in npz_files:
        episode = load_episode(npz_file)
        if "observation" not in episode:
            raise KeyError(f"Missing 'observation' in {npz_file}")

        observations = limited_observations(
            episode["observation"],
            total_observations,
            max_samples,
        )
        if len(observations) == 0:
            break

        if tuple(observations.shape[1:]) != tuple(rendered_shape):
            raise ValueError(
                f"Observation shape mismatch in {npz_file}: buffer has "
                f"{observations.shape[1:]}, rendered states have {rendered_shape}. "
                "Check obs_type, frame_stack, resolution, grayscale, and env config."
            )

        for observation in observations:
            key = observation_key(observation)
            state_idx = key_to_state.get(key)
            if state_idx is None:
                unmatched_counts[key] = unmatched_counts.get(key, 0) + 1
                unmatched_observations.setdefault(key, np.asarray(observation))
            else:
                state_counts[state_idx] += 1

        total_observations += len(observations)

    nearest_summary = {}
    if unmatched_observations:
        keys = list(unmatched_observations.keys())
        observations = np.stack([unmatched_observations[key] for key in keys], axis=0)
        nearest_indices, nearest_distances = nearest_rendered_state_indices(
            observations,
            rendered_observations,
            nearest_batch_size,
        )
        for key, state_idx in zip(keys, nearest_indices):
            state_counts[int(state_idx)] += unmatched_counts[key]

        nearest_summary = {
            "nearest_fallback_unique_observations": len(keys),
            "nearest_fallback_total_observations": int(
                sum(unmatched_counts.values())
            ),
            "nearest_max_squared_distance": float(np.max(nearest_distances)),
            "nearest_mean_squared_distance": float(np.mean(nearest_distances)),
        }

    return state_counts, total_observations, {
        "mode": "pixels",
        "rendered_states": n_states,
        **nearest_summary,
    }


def build_visitation_grid(env, state_counts):
    cells = env.unwrapped.cells
    max_x = max(cell[0] for cell in cells)
    max_y = max(cell[1] for cell in cells)
    min_x = min(cell[0] for cell in cells)
    min_y = min(cell[1] for cell in cells)

    grid_width = max_x - min_x + 1
    grid_height = max_y - min_y + 1
    grid = np.zeros((grid_height, grid_width), dtype=np.float64)

    for state_idx, count in enumerate(state_counts):
        state = env.unwrapped.idx_to_state[state_idx]
        x, y = state[:2]
        grid[y - min_y, x - min_x] += count

    return grid


def plot_heatmap(grid, total_observations, save_path):
    masked_grid = np.ma.masked_where(grid == 0, grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#B2B2B2")
    im = ax.imshow(masked_grid, cmap="YlOrRd", interpolation="nearest")
    ax.set_title(f"Replay Buffer State Visitation (n={total_observations})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax, label="Visit Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@hydra.main(config_path="configs", config_name="plot_replay_buffer_heatmap", version_base="1.1")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    replay_dir = Path(cfg.replay_buffer_dir).resolve()
    npz_files = sorted(replay_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {replay_dir}")

    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True) if hasattr(cfg, "env") else {}
    env_kwargs.pop("name", None)
    env = gym_env.make(
        cfg.task_name,
        cfg.obs_type,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
        seed=cfg.seed,
        resolution=cfg.resolution,
        grayscale=cfg.grayscale,
        url=False,
        **env_kwargs,
    )

    n_states = env.unwrapped.n_states
    max_samples = get_max_samples(cfg)
    first_episode = load_episode(npz_files[0])
    if "observation" not in first_episode:
        raise KeyError(f"Missing 'observation' in {npz_files[0]}")

    if np.asarray(first_episode["observation"]).ndim == 2:
        state_counts, total_observations, decode_info = count_one_hot_observations(
            npz_files,
            n_states,
            max_samples,
        )
    else:
        state_counts, total_observations, decode_info = count_pixel_observations(
            npz_files,
            env,
            max_samples,
            cfg.get("pixel_nearest_batch_size", 64),
        )

    grid = build_visitation_grid(env, state_counts)
    save_path = Path(cfg.output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(grid, total_observations, save_path)

    print(f"Loaded {len(npz_files)} episodes from {replay_dir}")
    print(f"Decoded {total_observations} observations using {decode_info['mode']} mode")
    if max_samples is not None:
        print(f"Stopped after first_n_elements={max_samples}")
    for key, value in decode_info.items():
        if key != "mode":
            print(f"{key}: {value}")
    print(f"Saved heatmap to {save_path.resolve()}")


if __name__ == "__main__":
    main()
