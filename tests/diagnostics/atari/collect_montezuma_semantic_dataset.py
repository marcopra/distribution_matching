"""Collect a compact, semantically stratified Montezuma transition dataset.

Dataset contains training observations plus ALE RAM metadata. Sampling balances
room/player-position/action bins and always retains salient events.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gym_env


def make_env(seed: int):
    return gym_env.make(
        "ALE/MontezumaRevenge-v5",
        "pixels",
        frame_stack=3,
        action_repeat=4,
        seed=seed,
        resolution=84,
        grayscale=True,
        url=True,
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=0.25,
        max_episode_steps=27000,
        atari={
            "score_mask": {"enabled": True, "band_height": 10, "color": 0},
            "terminal_on_life_loss": False,
        },
    )


def get_ram(env) -> np.ndarray:
    ale = getattr(env.unwrapped, "ale", None)
    if ale is None:
        raise RuntimeError("ALE interface unavailable through env.unwrapped.ale")
    return np.asarray(ale.getRAM(), dtype=np.uint8).copy()


def info_of(time_step) -> dict:
    info = getattr(time_step, "info", None)
    return info if isinstance(info, dict) else {}


def room_of(time_step, ram: np.ndarray, room_index: int) -> int:
    value = info_of(time_step).get("montezuma_room_id")
    return int(ram[room_index] if value is None else value)


class BalancedTransitionStore:
    def __init__(self, capacity: int, per_bin: int, xy_bin: int, rng):
        self.capacity = capacity
        self.per_bin = per_bin
        self.xy_bin = xy_bin
        self.rng = rng
        self.rows = []
        self.bin_rows = defaultdict(list)
        self.bin_seen = defaultdict(int)

    def _key(self, row):
        return (
            row["room"],
            row["x"] // self.xy_bin,
            row["y"] // self.xy_bin,
            row["action"],
        )

    def add(self, row, salient: bool):
        key = self._key(row)
        self.bin_seen[key] += 1
        slots = self.bin_rows[key]
        if salient or len(slots) < self.per_bin:
            self.rows.append(row)
            slots.append(len(self.rows) - 1)
        else:
            # Reservoir replacement within semantic bin.
            pick = int(self.rng.integers(self.bin_seen[key]))
            if pick < self.per_bin:
                self.rows[slots[pick]] = row

        if len(self.rows) > self.capacity:
            # Preserve salient transitions; randomly remove ordinary transition.
            candidates = [i for i, item in enumerate(self.rows) if not item["salient"]]
            remove = int(self.rng.choice(candidates)) if candidates else 0
            self.rows.pop(remove)
            self._reindex()

    def _reindex(self):
        self.bin_rows.clear()
        for index, row in enumerate(self.rows):
            self.bin_rows[self._key(row)].append(index)


def log_progress(worker_id, total_steps, target_steps, store, episode,
                 observed_rooms, event_counts, started_at):
    elapsed = max(time.monotonic() - started_at, 1e-6)
    print(
        f"[worker {worker_id}] collected={total_steps:,}/{target_steps:,} "
        f"({100 * total_steps / target_steps:5.1f}%) "
        f"retained={len(store.rows):,} episodes={episode:,} "
        f"rooms={dict(sorted(observed_rooms.items()))} "
        f"events={dict(event_counts)} rate={total_steps / elapsed:,.1f} steps/s",
        flush=True,
    )


def collect_worker(worker_id, args, target_steps, worker_capacity):
    seed = args.seed + worker_id
    rng = np.random.default_rng(seed)
    env = make_env(seed)
    store = BalancedTransitionStore(worker_capacity, args.per_bin, args.xy_bin, rng)
    episode = 0
    total_steps = 0
    event_counts = defaultdict(int)
    observed_rooms = defaultdict(int)
    started_at = time.monotonic()
    next_log = args.log_every

    try:
        while total_steps < target_steps:
            ts = env.reset()
            ram = get_ram(env)
            episode_step = 0
            while not ts.last() and total_steps < target_steps:
                action = int(rng.integers(env.action_space.n))
                next_ts = env.step(action)
                next_ram = get_ram(env)
                room = room_of(ts, ram, args.room_ram_index)
                next_room = room_of(next_ts, next_ram, args.room_ram_index)
                observed_rooms[room] += 1
                x, y = int(ram[args.x_ram_index]), int(ram[args.y_ram_index])
                nx, ny = int(next_ram[args.x_ram_index]), int(next_ram[args.y_ram_index])
                reward = float(next_ts.reward)
                terminal = bool(next_ts.last())

                events = []
                if next_room != room:
                    events.append("room_change")
                if reward != 0:
                    events.append("reward")
                if terminal:
                    events.append("terminal")
                if abs(nx - x) + abs(ny - y) >= args.motion_threshold:
                    events.append("large_motion")
                salient = bool(events)
                for event in events:
                    event_counts[event] += 1

                store.add(
                    {
                        "obs": np.asarray(ts.observation, dtype=np.uint8).copy(),
                        "next_obs": np.asarray(next_ts.observation, dtype=np.uint8).copy(),
                        "action": action,
                        "reward": reward,
                        "terminal": terminal,
                        # Unique across workers while preserving episode grouping.
                        "episode": worker_id * 1_000_000_000 + episode,
                        "episode_step": episode_step,
                        "ram": ram,
                        "next_ram": next_ram,
                        "room": room,
                        "next_room": next_room,
                        "x": x,
                        "y": y,
                        "next_x": nx,
                        "next_y": ny,
                        "salient": salient,
                        "event": "|".join(events) if events else "ordinary",
                    },
                    salient,
                )
                ts, ram = next_ts, next_ram
                episode_step += 1
                total_steps += 1
                if total_steps >= next_log and total_steps < target_steps:
                    log_progress(
                        worker_id, total_steps, target_steps, store, episode,
                        observed_rooms, event_counts, started_at,
                    )
                    next_log += args.log_every
            episode += 1
    finally:
        env.close()

    log_progress(
        worker_id, total_steps, target_steps, store, episode,
        observed_rooms, event_counts, started_at,
    )
    return {
        "worker_id": worker_id,
        "rows": store.rows,
        "steps": total_steps,
        "episodes": episode,
        "events": dict(event_counts),
        "observed_rooms": dict(observed_rooms),
    }


def collect(args):
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")

    step_targets = [
        args.env_steps // args.num_envs + (worker < args.env_steps % args.num_envs)
        for worker in range(args.num_envs)
    ]
    # Each worker retains extra candidates. Final merge rebalances globally.
    worker_capacity = max(
        args.per_bin,
        int(np.ceil(args.capacity / args.num_envs)) * 2,
    )
    print(
        f"Starting {args.num_envs} environment worker(s): "
        f"steps={args.env_steps:,}, final capacity={args.capacity:,}, "
        f"worker candidate capacity={worker_capacity:,}",
        flush=True,
    )

    if args.num_envs == 1:
        results = [collect_worker(0, args, step_targets[0], worker_capacity)]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.num_envs, mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    collect_worker, worker, args, step_targets[worker], worker_capacity
                )
                for worker in range(args.num_envs)
            ]
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[main] worker {result['worker_id']} finished; "
                    f"received {len(result['rows']):,} candidates",
                    flush=True,
                )

    rng = np.random.default_rng(args.seed)
    store = BalancedTransitionStore(args.capacity, args.per_bin, args.xy_bin, rng)
    event_counts = defaultdict(int)
    observed_rooms = defaultdict(int)
    total_steps = 0
    episodes = 0
    for result in sorted(results, key=lambda item: item["worker_id"]):
        total_steps += result["steps"]
        episodes += result["episodes"]
        for event, count in result["events"].items():
            event_counts[event] += count
        for room, count in result["observed_rooms"].items():
            observed_rooms[int(room)] += count
        rng.shuffle(result["rows"])
        for row in result["rows"]:
            store.add(row, bool(row["salient"]))

    if not store.rows:
        raise RuntimeError("No transitions collected")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = store.rows[0].keys()
    arrays = {key: np.asarray([row[key] for row in store.rows]) for key in keys}
    np.savez_compressed(output, **arrays)
    metadata = {
        "env_steps": total_steps,
        "saved_transitions": len(store.rows),
        "capacity": args.capacity,
        "retention_fraction": len(store.rows) / total_steps,
        "episodes": episodes,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "room_ram_index": args.room_ram_index,
        "x_ram_index": args.x_ram_index,
        "y_ram_index": args.y_ram_index,
        "xy_bin": args.xy_bin,
        "event_counts": dict(event_counts),
        "observed_room_transitions": {
            str(key): value for key, value in sorted(observed_rooms.items())
        },
        "rooms": {str(k): int(v) for k, v in zip(*np.unique(arrays["room"], return_counts=True))},
        "semantic_bins_retained": len(store.bin_rows),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/outputs/atari/montezuma_semantic.npz")
    parser.add_argument("--env-steps", type=int, default=1_000_000)
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Independent ALE processes collecting in parallel.")
    parser.add_argument("--log-every", type=int, default=10_000,
                        help="Log progress every N transitions per worker.")
    parser.add_argument("--per-bin", type=int, default=8)
    parser.add_argument("--xy-bin", type=int, default=4)
    parser.add_argument("--motion-threshold", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    # ALE Montezuma RAM map. Full RAM is saved so these can be changed later.
    parser.add_argument("--room-ram-index", type=int, default=3)
    parser.add_argument("--x-ram-index", type=int, default=42)
    parser.add_argument("--y-ram-index", type=int, default=43)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
