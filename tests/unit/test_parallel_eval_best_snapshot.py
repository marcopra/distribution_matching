import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from agent.utils_debug_visualization import CoverageProgress
from pretrain_parallel import Workspace


class ParallelEvalBestSnapshotTest(unittest.TestCase):
    def test_best_snapshot_overwrites_previous_file(self):
        self.addCleanup(self._cleanup_temp_dir)
        workspace = Workspace.__new__(Workspace)
        self.temp_dir = tempfile.TemporaryDirectory()
        workspace.work_dir = Path(self.temp_dir.name)
        workspace.cfg = SimpleNamespace(snapshot_dir="models", action_repeat=1)
        workspace.snapshot_steps = []
        workspace.save_snapshot_flag = False
        workspace.agent = SimpleNamespace()
        workspace._global_step = 10
        workspace._global_episode = 1
        workspace._coverage_progress = CoverageProgress()
        workspace._coverage_progress.update(12.0, 10)

        workspace.save_snapshot(filename="snapshot.pt", force=True)
        snapshot = workspace.work_dir / "models" / "snapshot.pt"
        first_size = snapshot.stat().st_size

        workspace._global_step = 20
        workspace._coverage_progress.update(18.0, 20)
        workspace.save_snapshot(filename="snapshot.pt", force=True)

        self.assertEqual(list(snapshot.parent.glob("*.pt")), [snapshot])
        self.assertGreater(snapshot.stat().st_size, 0)
        self.assertGreater(first_size, 0)
        payload = torch.load(snapshot, weights_only=False)
        self.assertEqual(payload["_global_step"], 20)
        self.assertEqual(payload["_coverage_progress"].best_coverage, 18.0)

    def _cleanup_temp_dir(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
