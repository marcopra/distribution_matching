#!/usr/bin/env python3
"""Reproduce the preserved image/state audit in an isolated source workspace.

Legacy mode uses audited_sources.zip, including the pre-fix actor implementation.
Current mode overlays current source and checks the corrected line-search branch.
Original checkpoints and audit artifacts are read only. CUDA is required by the
preserved checkpoint probes. Outputs default to tests/outputs/pointmaze/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "diagnostics/image_rover_20261007"


def prepare_workspace(output: Path, source: str) -> tuple[Path, Path]:
    workspace = output / "audit_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUDIT / "audited_sources.zip") as archive:
        for name in archive.namelist():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid archive path: {name}")
            # Run directories are linked read only below; never extract into them.
            if relative.parts[0] == "exp_local_parallel":
                continue
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source == "current" and (ROOT / relative).is_file():
                shutil.copy2(ROOT / relative, target)
            else:
                target.write_bytes(archive.read(name))

    for name in ("exp_local_parallel", "2606.21271v1.pdf", "Draft_Notes_for_Rover.pdf"):
        target = workspace / name
        if not target.exists():
            target.symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())

    artifacts = workspace / "diagnostics/image_rover_20261007"
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in ("audit.py", "supplement.py", "operator_probe.py"):
        shutil.copy2(AUDIT / name, artifacts / name)

    expected = json.loads((AUDIT / "environment.json").read_text())["source_sha256"]
    hashes = {name: hashlib.sha256((workspace / name).read_bytes()).hexdigest()
              for name in expected}
    if source == "legacy" and hashes != expected:
        mismatch = [name for name in expected if hashes[name] != expected[name]]
        raise RuntimeError(f"Archived audit source differs: {mismatch}")
    (output / "source_manifest.json").write_text(json.dumps({
        "source": source, "source_sha256": hashes,
        "original_audit_sha256": expected,
    }, indent=2) + "\n")
    return workspace, artifacts


def run_step(workspace: Path, artifacts: Path, script: str, stage: str | None) -> None:
    command = [sys.executable, str(artifacts / script)]
    if stage is not None:
        command.append(stage)
    label = script.removesuffix(".py") + (f"_{stage}" if stage else "")
    log = artifacts / f"{label}.log"
    print(f"Running {label}; log: {log}", flush=True)
    environment = dict(os.environ, MUJOCO_GL="egl")
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=workspace, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-6000:], file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)


def verify_numeric_results(reference, actual, path: str) -> int:
    """Check recorded numeric evidence, ignoring relocated paths and labels."""
    if isinstance(reference, dict):
        return sum(verify_numeric_results(value, actual[key], f"{path}.{key}")
                   for key, value in reference.items())
    if isinstance(reference, list):
        if len(reference) != len(actual):
            raise AssertionError(f"{path}: result length differs")
        return sum(verify_numeric_results(a, b, f"{path}[{index}]")
                   for index, (a, b) in enumerate(zip(reference, actual)))
    if isinstance(reference, bool):
        if reference != actual:
            raise AssertionError(f"{path}: boolean evidence differs")
    elif isinstance(reference, (int, float)):
        if not math.isclose(reference, actual, rel_tol=1e-6, abs_tol=1e-12):
            raise AssertionError(f"{path}: expected {reference}, got {actual}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("legacy", "current"), default="legacy")
    parser.add_argument("--full", action="store_true",
                        help="Also rerun all geometry, collapse, rollout, and operator probes.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "tests/outputs/pointmaze/image_rover_verification"
              / args.source).resolve()
    if output == AUDIT.resolve() or AUDIT.resolve() in output.parents:
        parser.error("Output must be separate from the original audit directory")
    output.mkdir(parents=True, exist_ok=True)
    workspace, artifacts = prepare_workspace(output, args.source)
    run_step(workspace, artifacts, "audit.py", "all" if args.full else "inventory")
    run_step(workspace, artifacts, "supplement.py", "line_search")
    run_step(workspace, artifacts, "supplement.py", "reproduce")
    if args.full:
        run_step(workspace, artifacts, "supplement.py", "geometry")
        run_step(workspace, artifacts, "supplement.py", "rollout_statistics")
        run_step(workspace, artifacts, "operator_probe.py", None)

    rows = json.loads((artifacts / "reproduced_evaluation.json").read_text())
    reference = {row["run"]: row for row in
                 json.loads((AUDIT / "reproduced_evaluation.json").read_text())}
    for row in rows:
        for metric, tolerance in (("return_mean", 1e-6), ("success_fraction", 1e-6),
                                  ("coverage_pct", 1e-4)):
            if not math.isclose(row[metric], reference[row["run"]][metric],
                                rel_tol=0., abs_tol=tolerance):
                raise AssertionError(f"{row['run']} {metric} failed reproduction: {row[metric]}")
    branch = json.loads((artifacts / "line_search_results.json").read_text())
    expected_acceptance = args.source == "legacy"
    if branch["accepted_rising_inner_iterate"] != expected_acceptance:
        raise AssertionError("Exhausted backtracking branch differs from expected behavior")
    full_numeric_values_verified = 0
    if args.full:
        for name in ("checkpoint_results.json", "supplement_geometry.json",
                     "operator_results.json", "algebra_results.json",
                     "evaluation_results.json", "paired_uncertainty.json"):
            full_numeric_values_verified += verify_numeric_results(
                json.loads((AUDIT / name).read_text()),
                json.loads((artifacts / name).read_text()), name,
            )
    summary = {
        "source": args.source, "verified": True, "full": args.full,
        "full_numeric_values_verified": full_numeric_values_verified,
        "artifacts": str(artifacts),
        "accepted_rising_inner_iterate": branch["accepted_rising_inner_iterate"],
        "evaluation": [{key: row[key] for key in
                        ("run", "frame", "episodes", "return_mean", "success_fraction", "coverage_pct")}
                       for row in rows],
        "scope": "Old checkpoints reproduce historical results; these are not post-fix training outcomes.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
