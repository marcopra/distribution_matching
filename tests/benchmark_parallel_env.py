import argparse
import multiprocessing as mp
import os
import sys
import statistics
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gym_env


def _sample_actions(env):
    return np.asarray([env.single_action_space.sample() for _ in range(env.num_envs)])


def run_env_only(task_name, obs_type, num_envs, total_transitions, warmup, repeats, env_kwargs):
    rates = []
    for repeat in range(repeats):
        env = gym_env.make_async_vector_env(
            num_envs,
            1000 + repeat * 100,
            task_name,
            obs_type,
            frame_stack=1,
            action_repeat=1,
            url=False,
            **env_kwargs,
        )
        try:
            env.reset(seed=[1000 + repeat * 100 + i for i in range(num_envs)])
            for _ in range(max(1, warmup // num_envs)):
                _, _, terminated, truncated, _ = env.step(_sample_actions(env))
                done = np.logical_or(terminated, truncated)
                if np.any(done):
                    env.reset(options={"reset_mask": done.astype(np.bool_)})
            vector_steps = max(1, total_transitions // num_envs)
            start = time.perf_counter()
            completed = 0
            for _ in range(vector_steps):
                _, _, terminated, truncated, _ = env.step(_sample_actions(env))
                done = np.logical_or(terminated, truncated)
                if np.any(done):
                    env.reset(options={"reset_mask": done.astype(np.bool_)})
                completed += num_envs
            elapsed = time.perf_counter() - start
            rates.append(completed / elapsed)
        finally:
            env.close()
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="CartPole-v1")
    parser.add_argument("--obs-type", default="states")
    parser.add_argument("--num-envs", default="1,2,4,8")
    parser.add_argument("--total-transitions", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args, unknown = parser.parse_known_args()
    env_kwargs = {}
    for item in unknown:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        env_kwargs[key.lstrip("-")] = value

    env_counts = [int(x) for x in args.num_envs.split(",") if x.strip()]
    env_counts = [x for x in env_counts if x <= max(1, os.cpu_count() or 1)]

    print(f"cpu_count={os.cpu_count()}")
    print(f"torch_threads={torch.get_num_threads()}")
    print(f"multiprocessing_start_method={mp.get_start_method(allow_none=True)}")
    print("device=cpu")
    print("mode=environment-only random actions")
    print()
    print("num_envs | transitions/s | speedup | efficiency")

    baseline = None
    for num_envs in env_counts:
        rates = run_env_only(
            args.task_name,
            args.obs_type,
            num_envs,
            args.total_transitions,
            args.warmup,
            args.repeats,
            env_kwargs,
        )
        mean_rate = statistics.mean(rates)
        stdev_rate = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        if baseline is None:
            baseline = mean_rate
        speedup = mean_rate / baseline if baseline else 1.0
        efficiency = speedup / num_envs
        print(f"{num_envs:8d} | {mean_rate:13.1f} +/- {stdev_rate:7.1f} | {speedup:7.2f} | {efficiency:10.2f}")


if __name__ == "__main__":
    main()
