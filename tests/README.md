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
