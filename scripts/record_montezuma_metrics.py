import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gym_env


def parse_actions(args):
    def parse_token(token):
        token = token.strip()
        if not token or token == "...":
            return None
        return int(token)

    if args.actions:
        return [value for value in (parse_token(token) for token in args.actions.split(",")) if value is not None]
    if args.actions_file:
        text = Path(args.actions_file).read_text().strip()
        return [value for value in (parse_token(token) for token in text.replace("\n", ",").split(",")) if value is not None]
    raise ValueError("Provide either --actions or --actions-file.")


def save_gif(frames, output_path, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    duration_ms = max(1, int(1000 / fps))
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def format_metrics(info):
    return {
        "room_id": info.get("montezuma_room_id"),
        "visited_second_room": info.get("montezuma_visited_second_room"),
        "max_room_id": info.get("montezuma_max_room_id"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=str, default=None,
                        help="Comma-separated discrete actions.")
    parser.add_argument("--actions-file", type=str, default=None,
                        help="Text file with comma- or newline-separated actions.")
    parser.add_argument("--output", type=str, default="debug/montezuma_metrics.gif")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-on-second-room", action="store_true")
    args = parser.parse_args()

    actions = parse_actions(args)
    if args.max_steps is not None:
        actions = actions[:args.max_steps]

    env = gym_env.make(
        "ALE/MontezumaRevenge-v5",
        obs_type="pixels",
        frame_stack=1,
        action_repeat=1,
        seed=args.seed,
        resolution=84,
        random_init=False,
        randomize_goal=False,
        url=True,
        score_mask=False,
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=0.0,
    )

    frames = []
    time_step = env.reset()
    frames.append(time_step.image_observation)

    prev_metrics = None
    reset_metrics = format_metrics(time_step.info or {})
    print(f"reset metrics: {reset_metrics}")

    for step_idx, action in enumerate(actions, start=1):
        time_step = env.step(action)
        frames.append(time_step.image_observation)

        metrics = format_metrics(time_step.info or {})
        if metrics != prev_metrics:
            print(
                f"step={step_idx:04d} action={action:02d} reward={time_step.reward:.2f} "
                f"done={time_step.last()} metrics={metrics}"
            )
            prev_metrics = metrics

        if args.stop_on_second_room and metrics["visited_second_room"]:
            print(f"Stopping at step {step_idx} after entering a new room.")
            break

        if time_step.last():
            print(f"Episode ended at step {step_idx}.")
            break

    output_path = Path(args.output)
    save_gif(frames, output_path, args.fps)
    env.close()
    print(f"Saved GIF to {output_path.resolve()}")


if __name__ == "__main__":
    main()
