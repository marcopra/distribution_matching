"""Inspect exact pixel observations used by Montezuma training."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gym_env


def main():
    output_dir = Path("montezuma_observation_smoke")
    output_dir.mkdir(exist_ok=True)
    env = gym_env.make(
        "ALE/MontezumaRevenge-v5",
        "pixels",
        frame_stack=3,
        action_repeat=4,
        seed=1,
        resolution=84,
        grayscale=True,
        url=True,
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=0.25,
        max_episode_steps=27000,
        atari={
            "score_mask": {
                "enabled": True,
                "band_height": 10,
                "color": 0,
            },
            "terminal_on_life_loss": False,
        },
    )
    try:
        print("Wrapper chain:")
        current = env
        while True:
            print(
                f"- {type(current).__module__}.{type(current).__name__}"
                f" band_height={getattr(current, 'band_height', 'N/A')}"
            )
            if not hasattr(current, "env"):
                break
            current = current.env

        time_step = env.reset()
        print(
            "Returned observation:",
            time_step.observation.shape,
            time_step.observation.dtype,
            f"range=[{time_step.observation.min()}, {time_step.observation.max()}]",
        )

        frames = [time_step.observation[-1].copy()]
        rng = np.random.default_rng(1)
        for _ in range(300):
            time_step = env.step(int(rng.integers(env.action_space.n)))
            frames.append(time_step.observation[-1].copy())
            if time_step.last():
                time_step = env.reset()

        frames = np.stack(frames)
        selected = [0, 10, 50, 100, 200, 300]
        contact_sheet = np.concatenate([frames[index] for index in selected], axis=1)
        Image.fromarray(contact_sheet).save(output_dir / "contact_sheet.png")
        for index in selected:
            Image.fromarray(frames[index]).save(output_dir / f"frame_{index:03d}.png")

        changed_pixel_mask = np.any(frames != frames[0], axis=0)
        changing_rows = np.flatnonzero(np.any(changed_pixel_mask, axis=1))
        print("Collected frames:", len(frames))
        print(
            "Byte-unique frames:",
            np.unique(frames.reshape(len(frames), -1), axis=0).shape[0],
        )
        print("Rows containing changes:", changing_rows.tolist())
        print(
            "Changed pixels per row:",
            np.count_nonzero(changed_pixel_mask, axis=1).tolist(),
        )
        print("Images saved to:", output_dir.resolve())
    finally:
        env.close()


if __name__ == "__main__":
    main()
