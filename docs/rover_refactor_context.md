# ROVER Refactor Context

This note gives future engineers the map of the ROVER refactor. The goal was to keep the ROVER algorithm files readable while making debugging plots and diagnostics reusable across variants.

## Module Layout

- `agent/rover.py`: canonical ROVER agent implementation.
- `agent/rover_nystrom.py`: production Nyström ROVER variant.
- `agent/rover_nystrom_pointmaze_debug.py`: PointMaze-oriented debug variant with configurable kernels.
- `agent/rover_nystrom_gridworld_debug.py`: gridworld-oriented debug variant.
- `agent/experimental/rover_nystrom_legacy_debug.py`: preserved legacy debug copy for reference only.
- `agent/rover_networks.py`: shared `Encoder`, `CNNEncoder`, and `ProjectSA`.
- `agent/rover_buffers.py`: shared encoded actor FIFO.
- `agent/rover_matchers.py`: shared distribution-matching math, using the richer kernel-aware matcher with inner-product defaults for production use.
- `agent/rover_visualization/`: shared exploration, gridworld, PointMaze, Fetch, trajectory, and visualizer-suite code.
- `agent/utils_debug_visualization.py`: compatibility shim that re-exports the new visualization modules.

## Adding A ROVER Variant

Start from `agent/rover.py` or `agent/rover_nystrom.py`, then import shared pieces from `agent.rover_networks`, `agent.rover_buffers`, `agent.rover_matchers`, and `agent.rover_visualization`. Do not copy visualization classes into the new algorithm file; wire the variant through `build_debug_visualizer_suite(...)` and add only variant-specific algorithm behavior.

Add a Hydra config under `configs/agent/` with the variant `_target_`, and update only the launchers or pretrain configs that should use the new variant.

## Moved Or Renamed Code

- `Encoder`, `CNNEncoder`, and `ProjectSA` moved from ROVER variant files to `agent/rover_networks.py`.
- `EncodedTransitionFIFO` moved from Nyström/debug variants to `agent/rover_buffers.py`.
- `DistributionMatcher` moved to `agent/rover_matchers.py`; the shared version keeps the debug kernel hooks and defaults to inner-product behavior.
- `FixedRandomEncoder`, `EmpiricalOccupancyTracker`, and `ExplorationVisualizer` moved to `agent/rover_visualization/exploration.py`.
- `DiscreteStateVisualizationAdapter` and `EmbeddingDistributionVisualizerV2` moved to `agent/rover_visualization/gridworld.py`.
- Domain and trajectory visualizers moved from `agent/utils_debug_visualization.py` to `agent/rover_visualization/domains.py`.
- `RoverDebugVisualizerSuite` and `build_debug_visualizer_suite` moved to `agent/rover_visualization/suite.py`.
- `agent/rover_nystrom_debug.py` became `agent/rover_nystrom_pointmaze_debug.py`.
- `agent/rover_nystrom_debug_gridworld.py` became `agent/rover_nystrom_gridworld_debug.py`.
- `agent/rover_nystrom_debug copy.py` became `agent/experimental/rover_nystrom_legacy_debug.py`.
- Root and `encoder_testing/` diagnostics moved under `tests/diagnostics/`; generated outputs now belong under `tests/outputs/`.
- Test configs moved from top-level `configs/config_test*.yaml` to `configs/test/`.

## Deleted Code

- Duplicate in-file definitions of shared ROVER networks, FIFO, matcher, and visualization classes were removed from the algorithm modules after moving them to shared modules. These were structural duplicates, not behavior removals.
- Tracked generated maze artifacts under `encoder_testing/maze_test_outputs/` were removed. The smoke diagnostic regenerates them under ignored `tests/outputs/gridworld/maze_test_outputs/`.
- No production ROVER class, config target, or launcher behavior was intentionally deleted.

## Validation Notes

- `python3 -m compileall agent tests` passes.
- `conda run -n dist_matching python -m unittest tests.unit.test_rover_encoded_actor_fifo` passes.
- `conda run -n dist_matching python tests/diagnostics/gridworld/smoke_maze_env.py` writes artifacts under `tests/outputs/gridworld/maze_test_outputs/`.
- `ruff` was not installed in the base interpreter or the `dist_matching` conda environment, so the dead-code pass used targeted `rg` checks plus compile/import validation.
