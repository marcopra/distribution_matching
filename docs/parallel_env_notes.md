# Parallel Environment Sampling Notes

## Autoreset and Terminal Observations

`pretrain_parallel.py` creates training envs with Gymnasium `AsyncVectorEnv` and `AutoresetMode.DISABLED`.
Done envs are reset explicitly with `reset(options={"reset_mask": done})`.

Terminal `ExtendedTimeStep` objects are stored before reset. The reset timestep returned by the reset mask is stored as the first timestep of that env's next replay episode, never as part of the terminated episode.

## Per-Environment Replay Storage

`replay_buffer_parallel.py` keeps one active episode accumulator per env. `add(..., env_id=i)` appends only to that env's accumulator. When one env reaches `LAST`, only that env's episode is finalized and written to disk.

Pending n-step transition streams are also built from the env-specific accumulator, so transitions from different workers cannot be interleaved.

Episode filenames include timestamp, nanosecond time, process ID, env ID, episode index, and episode length to avoid collisions.

## Step and Schedule Semantics

A vector step produces `num_envs` logical environment transitions. The loop uses scalar logical steps:

```text
logical_steps = global_step + np.arange(num_envs)
```

These logical environment steps are passed to `update_meta`, `act_parallel`, and `agent.update`. `global_step` advances by `num_envs` after storage.

## Update Ratio

Original pretraining performs one agent update per collected transition after seed warmup. The parallel loop preserves that ratio by running one update for each collected logical transition whose step is past seed warmup. The step argument passed into `agent.update` is the logical environment step, not an optimizer-call counter.

## Known Bottlenecks

Parallel speedup can be limited by inter-process communication, replay compression and disk writes, action inference on GPU/CPU, and optimizer updates. Use `tests/benchmark_parallel_env.py` to separate environment-only throughput from end-to-end training cost.
