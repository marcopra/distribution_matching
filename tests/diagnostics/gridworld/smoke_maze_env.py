"""Smoke test generated discrete maze environments.

Writes generated maze configs and a rendered PNG to
tests/outputs/gridworld/maze_test_outputs.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

import gym_env
from env.maze_generator import write_maze_files


def main() -> None:
    output_dir = Path("tests/outputs/gridworld/maze_test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_states = 500
    architecture_path = output_dir / f"maze_{n_states}_seed7.yaml"
    env_config_path = output_dir / f"maze_{n_states}_seed7_env.yaml"
    write_maze_files(
        n_states=n_states,
        seed=7,
        architecture_path=architecture_path,
        env_config_path=env_config_path,
        max_steps=600,
    )

    env = gym_env.make(
        "Maze-v0",
        obs_type="pixels",
        frame_stack=1,
        action_repeat=1,
        resolution=512,
        render_mode="rgb_array",
        maze_file=str(architecture_path),
        max_steps=600,
    )
    time_step = env.reset(seed=0)

    for action in [3, 3, 1, 1, 2, 0, 3, 1]:
        time_step = env.step(action)
        if time_step.last():
            break

    image = time_step.image_observation
    figure_path = output_dir / f"maze_{n_states}_seed7.png"
    Image.fromarray(image.astype("uint8")).save(figure_path)

    assert env.n_states == n_states
    assert image.shape == (512, 512, 3)
    print(f"Saved generated maze config to {architecture_path}")
    print(f"Saved env config to {env_config_path}")
    print(f"Saved rendered maze figure to {figure_path}")


if __name__ == "__main__":
    main()
