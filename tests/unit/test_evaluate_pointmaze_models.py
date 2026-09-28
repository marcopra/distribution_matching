import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluate_pointmaze_models import (
    EvaluationResult,
    PolicySpec,
    apply_expansion_status,
    coverage_statistics,
    discover_policy_specs,
    render_markdown,
    rover_kernel_metadata,
    run_evaluations,
    snapshot_identifier,
)


def make_spec(tmp: Path, *, step=None, snapshot_id="snapshot"):
    return PolicySpec(
        environment="umaze",
        observation_type="states",
        algorithm="rnd",
        algorithm_dir=tmp / "umaze/states/rnd",
        config_path=tmp / "umaze/states/rnd/.hydra/config.yaml",
        snapshot_path=None,
        snapshot_id=snapshot_id,
        numbered_step=step,
    )


class EvaluatePointMazeModelsTest(unittest.TestCase):
    def test_discovery_finds_every_snapshot_and_matrix_missing_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "umaze/states/rnd"
            (run / ".hydra").mkdir(parents=True)
            (run / ".hydra/config.yaml").touch()
            models = run / "models/nested"
            models.mkdir(parents=True)
            (models / "snapshot_20.pt").touch()
            (models / "snapshot_10.pt").touch()
            (models / "best_snapshot.pt").touch()
            (root / "umaze/states/random").mkdir(parents=True)

            specs = discover_policy_specs(root)
            rnd = [spec for spec in specs if spec.environment == "umaze" and spec.observation_type == "states" and spec.algorithm == "rnd"]
            self.assertEqual([spec.numbered_step for spec in rnd], [10, 20, None])
            self.assertEqual(len({spec.snapshot_id for spec in rnd}), 3)
            random_spec = next(spec for spec in specs if spec.environment == "umaze" and spec.observation_type == "states" and spec.algorithm == "random")
            self.assertEqual(random_spec.snapshot_id, "random")
            missing = next(spec for spec in specs if spec.environment == "umaze" and spec.observation_type == "pixels" and spec.algorithm == "rover")
            self.assertEqual(missing.snapshot_id, "missing")

    def test_snapshot_identifier_uses_relative_path(self):
        models = Path("/run/models")
        self.assertEqual(
            snapshot_identifier(models, models / "a b/seed_1/final_snapshot.pt"),
            "a_b__seed_1__final_snapshot",
        )

    def test_coverage_statistics_uses_standard_error(self):
        mean, se = coverage_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(mean, 3.0)
        self.assertAlmostEqual(se, 2.5 ** 0.5 / 5.0 ** 0.5)

    def test_expansion_only_compares_numbered_successful_snapshots(self):
        root = Path("/tmp/root")
        first = EvaluationResult(make_spec(root, step=100, snapshot_id="snapshot_100"), "ok", coverage_mean=10.0)
        second = EvaluationResult(make_spec(root, step=200, snapshot_id="snapshot_200"), "ok", coverage_mean=10.3)
        final = EvaluationResult(make_spec(root, snapshot_id="final_snapshot"), "ok", coverage_mean=99.0)
        apply_expansion_status([second, final, first], 0.25)
        self.assertIsNone(first.expanding)
        self.assertAlmostEqual(second.expansion_delta, 0.3)
        self.assertTrue(second.expanding)
        self.assertIsNone(final.expanding)

    def test_rover_kernel_metadata_prefers_active_snapshot_value(self):
        agent = SimpleNamespace(
            kernel_type="gaussian",
            kernel_bandwidth="linear(0.01, 0.18, 500000)",
            _active_kernel_bandwidth=0.18,
        )
        self.assertEqual(rover_kernel_metadata(agent, 1_000_000), ("gaussian", 0.18))
        scheduled = SimpleNamespace(
            kernel_type="gaussian",
            kernel_bandwidth="linear(0.01, 0.18, 500000)",
            _active_kernel_bandwidth=None,
        )
        self.assertEqual(rover_kernel_metadata(scheduled, 250_000), ("gaussian", 0.095))

    def test_run_continues_after_error_and_markdown_marks_status(self):
        root = Path("/tmp/root")
        bad = make_spec(root, snapshot_id="bad")
        good = make_spec(root, snapshot_id="good")

        def evaluator(spec):
            if spec.snapshot_id == "bad":
                raise RuntimeError("broken")
            return EvaluationResult(
                spec=spec,
                status="ok",
                step=10,
                horizon=500,
                coverage_values=[10.0] * 5,
                coverage_mean=10.0,
                coverage_se=0.0,
            )

        results, errors = run_evaluations([bad, good], evaluator)
        self.assertEqual([result.status for result in results], ["error", "ok"])
        self.assertIn("rnd", errors)
        args = argparse.Namespace(
            plot_trajectories=15,
            coverage_runs=5,
            coverage_trajectories=50,
            coverage_grid_size=90,
            coverage_radius=0.08,
            coverage_expansion_tolerance=0.25,
            coverage_seed=0,
            horizon_override=None,
        )
        markdown = render_markdown(results, args, root)
        self.assertIn("| bad | error | error |", markdown)
        self.assertIn("10.000, 10.000", markdown)


if __name__ == "__main__":
    unittest.main()
