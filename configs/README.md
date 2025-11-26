# Hydra Configuration for Distribution Matching

This directory contains modular Hydra configurations for the distribution matching experiments.

## Structure

```
configs/
├── config.yaml          # Main configuration file
├── env/                 # Environment-specific configurations
│   ├── single_room.yaml
│   ├── two_rooms.yaml
│   └── four_rooms.yaml
└── README.md
```

## Usage

### Basic Usage

Run with default configuration (SingleRoom environment):
```bash
python distribution_matching.py
```

### Override Environment

Run with TwoRooms environment:
```bash
python distribution_matching.py env=two_rooms
```

Run with FourRooms environment:
```bash
python distribution_matching.py env=four_rooms
```

### Override Parameters

Override specific hyperparameters:
```bash
python distribution_matching.py eta=0.01 gamma=0.99
```

Override environment size:
```bash
python distribution_matching.py env=two_rooms env.room_size=7
```

### Multiple Overrides

Combine multiple overrides:
```bash
python distribution_matching.py env=four_rooms n_updates=200000 eta=0.005
```

### Set Output Directory

Change output directory:
```bash
python distribution_matching.py output.save_dir=/path/to/output
```

## Configuration Files

### Main Config (`config.yaml`)

Contains:
- **defaults**: Specifies which environment config to use by default
- **experiment**: Hyperparameters for distribution matching
  - `gamma`: Discount factor (default: 0.95)
  - `eta`: Learning rate for mirror descent (default: 0.001)
  - `alpha`: Alpha parameter for KL divergence (default: 0.1)
  - `gradient_type`: 'reverse' or 'forward' (default: 'reverse')
  - `n_updates`: Number of optimization iterations (default: 100000)
  - `print_every`: Save interval (default: 20000)
  - `n_rollouts`: Number of rollout episodes (default: 1)
  - `horizon`: Rollout horizon (default: 9)
  - `initial_mode`: Initial distribution mode (default: 'top_left_cell')
  - `seed`: Random seed (default: 42)
- **output**: Output file paths

### Environment Configs (`env/*.yaml`)

Each environment config contains:
- `name`: Gymnasium environment ID
- `room_size`: Size of each room
- `max_steps`: Maximum episode steps
- `render_mode`: Rendering mode (null, 'human', 'rgb_array')
- `show_coordinates`: Show coordinate axes in rendering
- `goal_position`: Fixed goal position or null for random
- `start_position`: Fixed start position or null for random

Environment-specific parameters:
- **TwoRooms**: `corridor_length`, `corridor_y`
- **FourRooms**: `corridor_length`, `corridor_positions` (horizontal, vertical)

## Examples

### Example 1: Quick test with smaller updates
```bash
python distribution_matching.py n_updates=10000 print_every=2000
```

### Example 2: Two rooms with different corridor
```bash
python distribution_matching.py env=two_rooms env.corridor_y=3 env.room_size=6
```

### Example 3: Four rooms with custom learning rate
```bash
python distribution_matching.py env=four_rooms eta=0.002 gamma=0.98
```

### Example 4: Different initial distribution
```bash
python distribution_matching.py initial_mode=uniform
```

Available initial modes:
- `top_left_cell`: Concentrated in top-left cell
- `corner`: Distributed in corner region
- `uniform`: Uniform distribution
- `left_room`: Concentrated in left room (TwoRooms only)
- `corridor`: Concentrated in corridor (TwoRooms only)

## Advanced Usage

### Using Hydra Multirun

Run experiments with different parameters:
```bash
python distribution_matching.py -m env=single_room,two_rooms,four_rooms
```

Run hyperparameter sweep:
```bash
python distribution_matching.py -m eta=0.001,0.01,0.1 gamma=0.9,0.95,0.99
```

### Hydra Working Directory

By default, Hydra creates output directories in `outputs/`. Override with:
```bash
python distribution_matching.py hydra.run.dir=/path/to/output
```

Or disable it:
```bash
python distribution_matching.py hydra.output_subdir=null
```
