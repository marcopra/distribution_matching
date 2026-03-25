import argparse
from pathlib import Path
import sys

import numpy as np
import pygame
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gym_env


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


def build_action_lookup(action_meanings):
    return {name: idx for idx, name in enumerate(action_meanings)}


def current_action(action_lookup):
    keys = pygame.key.get_pressed()
    want_fire = keys[pygame.K_SPACE]
    direction = None

    if keys[pygame.K_UP]:
        direction = "UP"
    elif keys[pygame.K_DOWN]:
        direction = "DOWN"
    elif keys[pygame.K_LEFT]:
        direction = "LEFT"
    elif keys[pygame.K_RIGHT]:
        direction = "RIGHT"

    if want_fire and direction is not None:
        combo = f"{direction}FIRE"
        if combo in action_lookup:
            return action_lookup[combo]

    if want_fire and "FIRE" in action_lookup:
        return action_lookup["FIRE"]

    if direction is not None and direction in action_lookup:
        return action_lookup[direction]

    return action_lookup.get("NOOP", 0)


def blit_frame(screen, frame):
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    scaled = pygame.transform.scale(frame_surface, screen.get_size())
    screen.blit(scaled, (0, 0))
    pygame.display.flip()


def save_actions(actions, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(",".join(str(action) for action in actions) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif-output", type=str, default="debug/montezuma_keyboard.gif")
    parser.add_argument("--actions-output", type=str, default="debug/montezuma_keyboard_actions.txt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--stop-on-second-room", action="store_true")
    args = parser.parse_args()

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

    action_meanings = env.unwrapped.get_action_meanings()
    action_lookup = build_action_lookup(action_meanings)

    print("Action meanings:")
    for idx, meaning in enumerate(action_meanings):
        print(f"  {idx}: {meaning}")
    print("Controls: arrows move, space adds FIRE/jump, r resets, q quits and saves.")

    pygame.init()
    clock = pygame.time.Clock()

    time_step = env.reset()
    frame = time_step.image_observation
    height, width = frame.shape[:2]
    screen = pygame.display.set_mode((width * args.scale, height * args.scale))
    pygame.display.set_caption("Montezuma Keyboard Recorder")

    frames = [frame]
    actions = []
    prev_metrics = format_metrics(time_step.info or {})
    print(f"reset metrics: {prev_metrics}")

    running = True
    step_idx = 0
    while running and step_idx < args.max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    time_step = env.reset()
                    frame = time_step.image_observation
                    frames.append(frame)
                    actions.append(action_lookup.get("NOOP", 0))
                    prev_metrics = format_metrics(time_step.info or {})
                    print(f"manual reset metrics: {prev_metrics}")

        action = current_action(action_lookup)
        time_step = env.step(action)
        frame = time_step.image_observation

        actions.append(action)
        frames.append(frame)
        step_idx += 1

        metrics = format_metrics(time_step.info or {})
        if metrics != prev_metrics:
            print(
                f"step={step_idx:04d} action={action:02d} meaning={action_meanings[action]} "
                f"reward={time_step.reward:.2f} done={time_step.last()} metrics={metrics}"
            )
            prev_metrics = metrics

        blit_frame(screen, frame)

        if args.stop_on_second_room and metrics["visited_second_room"]:
            print(f"Stopping at step {step_idx} after entering a new room.")
            break

        if time_step.last():
            print(f"Episode ended at step {step_idx}. Resetting environment.")
            time_step = env.reset()
            frame = time_step.image_observation
            frames.append(frame)
            prev_metrics = format_metrics(time_step.info or {})
            print(f"reset metrics: {prev_metrics}")

        clock.tick(args.fps)

    pygame.quit()
    env.close()

    gif_output = Path(args.gif_output)
    actions_output = Path(args.actions_output)
    save_gif(frames, gif_output, args.fps)
    save_actions(actions, actions_output)

    print(f"Saved GIF to {gif_output.resolve()}")
    print(f"Saved actions to {actions_output.resolve()}")


if __name__ == "__main__":
    main()
