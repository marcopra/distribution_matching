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
    state_counts = np.zeros(n_states, dtype=np.int64)
    total_observations = 0

    for npz_file in npz_files:
        episode = load_episode(npz_file)
        if "observation" not in episode:
            raise KeyError(f"Missing 'observation' in {npz_file}")

        state_indices = decode_state_indices(episode["observation"])
        if np.any(state_indices >= n_states):
            raise ValueError(
                f"Found decoded states outside the environment range in {npz_file}: "
                f"max index {state_indices.max()}, env has {n_states} states"
            )

        state_counts += np.bincount(state_indices, minlength=n_states)
        total_observations += len(state_indices)

    grid = build_visitation_grid(env, state_counts)
    save_path = Path(cfg.output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(grid, total_observations, save_path)

    print(f"Loaded {len(npz_files)} episodes from {replay_dir}")
    print(f"Decoded {total_observations} one-hot observations")
    print(f"Saved heatmap to {save_path.resolve()}")


if __name__ == "__main__":
    main()
