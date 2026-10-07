# Verify the image ROVER fixes

The actor now retains raw observations for both modalities, re-encodes sampled
support under one encoder snapshot before landmark selection, and keeps that
snapshot fixed between actor fits. Pixel observations stay as CPU uint8. Failed
backtracking rejects worsening/nonfinite candidates and stops at the last
accepted policy. `actor_eta` follows the policy actually returned by best-iterate
restoration; `actor_backtracking_rejections` records exhausted searches.

Start from the repository root:

```bash
cd /home/mprattico-iit.local/distribution_matching
export ROVER_VERIFY_PYTHON=/home/mprattico-iit.local/miniconda3/envs/dist_matching/bin/python
export MUJOCO_GL=egl
```

## 1. Check implementation correctness

```bash
"$ROVER_VERIFY_PYTHON" -m pytest -q \
  tests/unit/test_rover_actor_encoder_drift.py \
  tests/unit/test_rover_pmd_backtracking.py \
  tests/unit/test_rover_sink_kernel.py \
  tests/unit/test_rover_pixel_diagnostic_coordinates.py \
  tests/unit/test_rover_pointmaze_whitening.py \
  tests/unit/test_rover_state_action_kernel.py \
  tests/unit/test_pretrain_eval_snapshots.py

"$ROVER_VERIFY_PYTHON" tests/diagnostics/pointmaze/smoke_image_rover_fixes.py
```

Expected: 40 tests pass. The smoke check uses real environments and the experiment
encoders with 32 support rows/16 requested landmarks. Both modalities report
`verified=true`, zero support mismatch and zero action-probability drift after an
encoder update and replay drain. Frozen coverage weights must remain unchanged,
and its cached features must encode the newest frame. The smoke check disables
TF32 for strict comparisons across convolution batch sizes.

Smoke output: `tests/outputs/pointmaze/image_rover_verification/smoke/results.json`.
These small fits do not measure training performance.

## 2. Reproduce the previous audit

```bash
"$ROVER_VERIFY_PYTHON" tests/diagnostics/pointmaze/reproduce_image_rover_audit.py --source legacy
```

This verifies the archived source hashes, reproduces the old failed-line-search
branch, and replays the original 50-episode evaluation at 100k frames:

| Metric | Image | State |
| --- | ---: | ---: |
| Coverage | 84.21516755% | 79.67372134% |
| Initial-goal success | 72% | 84% |
| Mean retargeted return | 1.20 | 1.28 |

To regenerate the complete representation, geometry, rollout, sink-algebra and
signed-operator evidence and compare 1,234 numeric values against the audit, add
`--full`:

```bash
"$ROVER_VERIFY_PYTHON" tests/diagnostics/pointmaze/reproduce_image_rover_audit.py --source legacy --full
```

To verify old-checkpoint inference compatibility and the corrected backtracking
branch under current code:

```bash
"$ROVER_VERIFY_PYTHON" tests/diagnostics/pointmaze/reproduce_image_rover_audit.py --source current
```

The replay summary is saved under
`tests/outputs/pointmaze/image_rover_verification/{legacy,current}/summary.json`.
Detailed artifacts and logs live in each summary's `artifacts` directory. Legacy
backtracking reports `accepted_rising_inner_iterate=true`; current reports
`false`. Both replay the same historical policies. Original audit evidence and
checkpoint files are preserved.

## 3. Run new training to measure post-fix behavior

Check both resolved Hydra commands without starting training:

```bash
ROVER_VERIFY_DRY_RUN=1 bash launchers/local/pointmaze/verify_rover_image_fixes.sh 1
```

Run one paired image/state comparison through the original 100k horizon:

```bash
ROVER_VERIFY_FRAMES=100000 bash launchers/local/pointmaze/verify_rover_image_fixes.sh 1
```

For the planned 200k comparison with paired seeds 1, 2, 3:

```bash
bash launchers/local/pointmaze/verify_rover_image_fixes.sh
```

Each command creates fresh `exp_local_parallel/..._fixed_{image,state}_s{seed}`
directories. Runs execute sequentially on one GPU, use 50 coverage trajectories
per evaluation every 20k frames, and save 50k/100k/200k checkpoints when reached.
The image run retains its three-frame dynamics input and newest-frame frozen
coverage CNN. The original modality-specific encoder schedules and landmark
strategies are retained; both runs use InfoNCE, the original sink ramp, and
`lambda_reg=1e-6`.

Inspect `tb/` or `pretrain.log` for coverage and retargeted return, and `tb/` for
`train/actor_backtracking_rejections`, `train/actor_eta`, and `train/actor_loss`.
A rejection can occur when the estimated direction cannot reduce the fixed
objective. It must preserve the last accepted policy. Training's evaluation log
does not record initial-goal success; the historical replay above computes it
from episode outcomes.

Start fresh for pixel training: old pixel FIFO checkpoints lack the observations
needed to rebuild historical features and now raise a clear error on resume.
They remain usable for evaluation. At capacity 100k, raw uint8 observation pairs
add about 4.23 GB of CPU storage before features and sampling temporaries; saved
pixel checkpoints are correspondingly larger.

Fresh training is needed to measure any performance gain. The old single-seed
comparison and small smoke check do not establish that the bugs caused the
success gap. A causal training ablation would additionally need paired legacy
runs with the corrected line-search guard held fixed.

## Validation scope

The 40 focused tests, both real-environment smoke modes, legacy/current historical
evaluation replay, the full legacy audit and both launcher configurations were
checked after the fixes.
Broader repository testing still has pre-existing renamed-module imports,
obsolete Montezuma blockwise/fresh-replay API tests, and parallel image-counter
expectations. The latter failures also reproduce against the archived pre-fix
source. They are outside these three fixes.
