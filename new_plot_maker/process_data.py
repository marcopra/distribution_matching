"""Align downloaded WandB CSV files onto a common x-grid for plotting."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

try:
    from .utils import Logger
except ImportError:
    from utils import Logger


METADATA_COLUMNS = ["__run_id", "__run_name", "__group", "__tags"]


def is_run_csv(path: Path) -> bool:
    return path.suffix == ".csv" and path.name not in {"runs_metadata.csv"}


def discover_leaf_folders(root: Path) -> list[Path]:
    leaves = []
    for folder in sorted([root, *[p for p in root.rglob("*") if p.is_dir()]]):
        if any(is_run_csv(path) for path in folder.iterdir() if path.is_file()):
            leaves.append(folder)
    return leaves


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(frame.columns)}")
    return pd.to_numeric(frame[column], errors="coerce")


def make_grid(frames: list[pd.DataFrame], x_key: str, cfg: DictConfig) -> np.ndarray:
    mins = []
    maxs = []
    for frame in frames:
        x = numeric_series(frame, x_key).dropna()
        if not x.empty:
            mins.append(float(x.min()))
            maxs.append(float(x.max()))

    if not mins:
        raise ValueError(f"No finite values found for x key '{x_key}'")

    grid_min = float(cfg.grid.min_x) if cfg.grid.min_x is not None else min(mins)
    grid_max = float(cfg.grid.max_x) if cfg.grid.max_x is not None else max(maxs)
    if cfg.grid.overlap_only:
        grid_min = float(cfg.grid.min_x) if cfg.grid.min_x is not None else max(mins)
        grid_max = float(cfg.grid.max_x) if cfg.grid.max_x is not None else min(maxs)
    if grid_max <= grid_min:
        raise ValueError(f"Invalid grid range: min={grid_min}, max={grid_max}")
    return np.linspace(grid_min, grid_max, int(cfg.grid.n_points))


def align_frame(
    frame: pd.DataFrame,
    grid: np.ndarray,
    x_key: str,
    y_keys: list[str],
    extrapolate: bool,
) -> pd.DataFrame:
    sorted_frame = frame.copy()
    sorted_frame[x_key] = numeric_series(sorted_frame, x_key)
    sorted_frame = sorted_frame.dropna(subset=[x_key]).sort_values(x_key)
    sorted_frame = sorted_frame.drop_duplicates(subset=[x_key], keep="last")

    aligned = pd.DataFrame({x_key: grid})
    for y_key in y_keys:
        y = numeric_series(sorted_frame, y_key)
        valid = sorted_frame[x_key].notna() & y.notna()
        x_values = sorted_frame.loc[valid, x_key].to_numpy(dtype=float)
        y_values = y.loc[valid].to_numpy(dtype=float)
        if len(x_values) < 2:
            aligned[y_key] = np.nan
            continue

        aligned_values = np.interp(grid, x_values, y_values, left=np.nan, right=np.nan)
        if extrapolate and len(x_values) >= 2:
            left_mask = grid < x_values[0]
            right_mask = grid > x_values[-1]
            left_slope = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
            right_slope = (y_values[-1] - y_values[-2]) / (x_values[-1] - x_values[-2])
            aligned_values[left_mask] = y_values[0] + left_slope * (grid[left_mask] - x_values[0])
            aligned_values[right_mask] = y_values[-1] + right_slope * (
                grid[right_mask] - x_values[-1]
            )
        aligned[y_key] = aligned_values

    for column in METADATA_COLUMNS:
        if column in frame.columns:
            aligned[column] = frame[column].dropna().iloc[0] if not frame[column].dropna().empty else ""
    return aligned


def log_input_frame_debug(
    csv_path: Path,
    frame: pd.DataFrame,
    x_key: str,
    y_keys: list[str],
) -> None:
    x = numeric_series(frame, x_key)
    finite_x = x.dropna()
    if finite_x.empty:
        Logger.detail(f"{csv_path.name}: no finite x values", color="yellow")
        return

    duplicate_x = int(x.duplicated().sum())
    monotonic = bool(finite_x.is_monotonic_increasing)
    y_summaries = []
    for y_key in y_keys:
        if y_key in frame.columns:
            finite_y = numeric_series(frame, y_key).notna().sum()
            y_summaries.append(f"{y_key}: {finite_y}/{len(frame)} finite")
        else:
            y_summaries.append(f"{y_key}: missing")

    Logger.detail(
        f"{csv_path.name}: rows={len(frame)}, "
        f"x=[{float(finite_x.min()):.4g}, {float(finite_x.max()):.4g}], "
        f"finite_x={len(finite_x)}/{len(frame)}, "
        f"duplicate_x={duplicate_x}, monotonic={monotonic}"
    )
    Logger.detail("  " + "; ".join(y_summaries))


def relative_output_path(input_root: Path, output_root: Path, csv_path: Path) -> Path:
    return output_root / csv_path.relative_to(input_root)


def process_folder(folder: Path, input_root: Path, output_root: Path, cfg: DictConfig) -> int:
    csv_paths = sorted(path for path in folder.iterdir() if path.is_file() and is_run_csv(path))
    if not csv_paths:
        return 0

    frames = [pd.read_csv(path) for path in csv_paths]
    grid = make_grid(frames, cfg.data.x_key, cfg)
    debug = bool(cfg.get("debug", False))

    Logger.subsection(str(folder.relative_to(input_root) or "."))
    Logger.item(f"files: {len(csv_paths)}")
    Logger.item(
        f"grid: {len(grid)} points from {grid[0]:.4g} to {grid[-1]:.4g}",
        color="green",
    )
    if debug:
        Logger.detail("input diagnostics:")
        for csv_path, frame in zip(csv_paths, frames):
            log_input_frame_debug(
                csv_path=csv_path,
                frame=frame,
                x_key=cfg.data.x_key,
                y_keys=list(cfg.data.y_keys),
            )

    written = 0
    for csv_path, frame in zip(csv_paths, frames):
        aligned = align_frame(
            frame=frame,
            grid=grid,
            x_key=cfg.data.x_key,
            y_keys=list(cfg.data.y_keys),
            extrapolate=bool(cfg.grid.extrapolate),
        )
        output_path = relative_output_path(input_root, output_root, csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aligned.to_csv(output_path, index=False)
        Logger.detail(f"saved: {output_path}")
        written += 1

    metadata_path = folder / "runs_metadata.csv"
    if metadata_path.exists():
        output_metadata = relative_output_path(input_root, output_root, metadata_path)
        output_metadata.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(metadata_path).to_csv(output_metadata, index=False)

    return written


def process_data(cfg: DictConfig) -> None:
    input_root = Path(cfg.data.input_dir)
    output_root = Path(cfg.data.output_dir or cfg.data.input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    Logger.section("Processing")
    Logger.item(f"input: {input_root}")
    Logger.item(f"output: {output_root}")
    Logger.item(f"x key: {cfg.data.x_key}")
    Logger.item(f"y keys: {', '.join(cfg.data.y_keys)}")

    leaf_folders = discover_leaf_folders(input_root)
    Logger.item(f"folders with CSV files: {len(leaf_folders)}", color="green")

    total_written = 0
    for folder in leaf_folders:
        total_written += process_folder(folder, input_root, output_root, cfg)

    Logger.item(f"processed files: {total_written}", color="green")


@hydra.main(version_base=None, config_path="configs", config_name="process")
def main(cfg: DictConfig) -> None:
    Logger.section("Configuration")
    Logger.detail(OmegaConf.to_yaml(cfg).rstrip())
    process_data(cfg)


if __name__ == "__main__":
    main()
