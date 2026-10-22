# Tests And Diagnostics

This directory keeps automated tests and exploratory diagnostics separate.

## Layout

- `tests/unit/`: automated unit tests that should be safe to run repeatedly.
- `tests/diagnostics/pointmaze/`: PointMaze rendering, policy, and kernel diagnostics.
- `tests/diagnostics/gridworld/`: gridworld/MultipleRooms dataset, maze, and encoder-sweep diagnostics.
- `tests/diagnostics/encoders/`: saved-encoder evaluation scripts.
- `tests/diagnostics/atari/`: Atari/Pong salient-episode utilities.
- `tests/outputs/`: generated plots, videos, datasets, summaries, and temporary configs. This directory is ignored by Git.

## Commands

Run the unit tests:

```bash
conda run -n dist_matching python -m unittest tests.unit.test_rover_encoded_actor_fifo
```

Run the lightweight gridworld smoke diagnostic:

```bash
conda run -n dist_matching python tests/diagnostics/gridworld/smoke_maze_env.py
```

Run a PointMaze policy diagnostic:

```bash
conda run -n dist_matching python tests/diagnostics/pointmaze/evaluate_pointmaze_policy.py \
  --snapshot exp_local/.../snapshot.pt \
  --output-dir tests/outputs/pointmaze/policy_eval
```

All new diagnostics should default to writing under `tests/outputs/`.

## Streamed Nyström memory benchmark

Run production-size float64 actor algebra on CUDA, without environment training:

```bash
python tests/diagnostics/pointmaze/benchmark_nystrom_memory.py
```

Defaults: 60,004 and 80,001 support rows, 15,000 landmarks, 1,024 dynamics
features, projection rank 1,000, and kernel row blocks of 1,024. Reports peak
CUDA allocated/reserved memory, peak host RSS, runtime, and live CUDA allocation
after release under `tests/outputs/pointmaze/`. Asserts peak allocation below
20 GiB. This synthetic benchmark complements the full PointMaze training run.
