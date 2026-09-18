import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from agent.utils_debug_visualization import CoverageProgress
from pretrain import Workspace


class PretrainEvalSnapshotsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace = Workspace.__new__(Workspace)
        self.workspace.work_dir = Path(self.temp_dir.name)
        self.workspace.cfg = SimpleNamespace(snapshot_dir="models", action_repeat=1)
        self.workspace.snapshot_steps = []
        self.workspace.save_snapshot_flag = False
        self.workspace.agent = SimpleNamespace()
        self.workspace._global_step = 10
        self.workspace._global_episode = 1
        self.workspace._coverage_progress = CoverageProgress()
        self.workspace._coverage_progress.update(12.0, 10)

    def test_named_best_and_final_snapshots_are_saved_when_periodic_saving_is_off(self):
        self.workspace.save_snapshot(filename="best_snapshot.pt", force=True)
        self.workspace._global_step = 20
        self.workspace.save_snapshot(filename="final_snapshot.pt", force=True)

        snapshot_dir = self.workspace.work_dir / "models"
        self.assertEqual(
            sorted(path.name for path in snapshot_dir.glob("*.pt")),
            ["best_snapshot.pt", "final_snapshot.pt"],
        )
        best = torch.load(snapshot_dir / "best_snapshot.pt", weights_only=False)
        final = torch.load(snapshot_dir / "final_snapshot.pt", weights_only=False)
        self.assertEqual(best["_global_step"], 10)
        self.assertEqual(final["_global_step"], 20)
        self.assertEqual(best["_coverage_progress"].best_coverage, 12.0)

    def test_empty_periodic_schedule_does_not_index_empty_list(self):
        self.workspace.save_snapshot()
        self.assertFalse((self.workspace.work_dir / "models" / "snapshot.pt").exists())


if __name__ == "__main__":
    unittest.main()
