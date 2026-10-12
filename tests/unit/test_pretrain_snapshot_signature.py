import json
from pathlib import Path
from types import SimpleNamespace

from pretrain_parallel import Workspace


def make_workspace(snapshot_dir):
    workspace = Workspace.__new__(Workspace)
    workspace.work_dir = Path(snapshot_dir)
    workspace.device = "cpu"
    workspace.cfg = SimpleNamespace(
        task_name="PointMaze_UMaze-v3",
        obs_type="pixels",
        frame_stack=3,
        grayscale=True,
        discount=0.99,
        nstep=1,
        p_path="none",
        p_path_coverage="none",
    )
    workspace.agent = SimpleNamespace(
        obs_shape=(9, 84, 84),
        dynamics_kernel_input="states",
        coverage_kernel_input="pixels",
        coverage_state_indices=(),
        coverage_frame_index=2,
        proprio_state_dims=4,
        feature_dim=512,
        kernel_type="gaussian",
        kernel_bandwidth_mult=1.0,
        coverage_kernel_bandwidth=None,
        coverage_kernel_bandwidth_mult=1.0,
    )
    return workspace


def test_legacy_checkpoint_without_signature_is_skipped_before_torch_load(tmp_path):
    workspace = make_workspace(tmp_path)
    original_agent = workspace.agent
    (tmp_path / "snapshot.pt").write_bytes(b"not a torch checkpoint")

    assert workspace.load_snapshot() is False
    assert workspace.agent is original_agent


def test_mismatched_checkpoint_signature_is_skipped_before_torch_load(tmp_path):
    workspace = make_workspace(tmp_path)
    original_agent = workspace.agent
    (tmp_path / "snapshot.pt").write_bytes(b"not a torch checkpoint")
    signature_path = tmp_path / "snapshot.pt.signature.json"
    signature_path.write_text(json.dumps({"task_name": "OtherMaze"}))

    assert workspace.load_snapshot() is False
    assert workspace.agent is original_agent
