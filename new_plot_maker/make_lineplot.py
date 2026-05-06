"""Create one or many seaborn lineplots from processed WandB CSV folders."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

try:
    from .utils import Logger
except ImportError:
    from utils import Logger


def is_run_csv(path: Path) -> bool:
    return path.suffix == ".csv" and path.name not in {"runs_metadata.csv"}


def contains_csv(path: Path) -> bool:
    return any(is_run_csv(candidate) for candidate in path.rglob("*.csv"))


def read_title(path: Path, configured_titles: dict[str, str] | None = None) -> str:
    configured_titles = configured_titles or {}
    if path.name in configured_titles:
        return configured_titles[path.name]
    title_path = path / "title.txt"
    if title_path.exists():
        title = title_path.read_text(encoding="utf-8").strip()
        if title:
            return title
    return path.name


def discover_plot_roots(
    input_dir: Path,
    mode: str,
    configured_titles: dict[str, str] | None = None,
    panel_order: list[str] | None = None,
) -> dict[str, Path]:
    direct_csv = any(is_run_csv(path) for path in input_dir.iterdir() if path.is_file())
    child_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())

    if mode == "single" or (mode == "auto" and direct_csv):
        return {read_title(input_dir, configured_titles): input_dir}

    candidate_roots = {path.name: path for path in child_dirs if contains_csv(path)}
    ordered_names = []
    for name in panel_order or []:
        if name in candidate_roots:
            ordered_names.append(name)
        else:
            Logger.item(f"panel_order entry not found or has no CSV files: {name}", color="yellow")
    ordered_names.extend(name for name in candidate_roots if name not in ordered_names)

    roots = {
        read_title(candidate_roots[name], configured_titles): candidate_roots[name]
        for name in ordered_names
    }
    if roots:
        return roots
    if contains_csv(input_dir):
        return {read_title(input_dir, configured_titles): input_dir}
    raise FileNotFoundError(f"No run CSV files found below {input_dir}")


def infer_group_name(csv_path: Path, frame: pd.DataFrame) -> str:
    if "__group" in frame.columns and not frame["__group"].dropna().empty:
        return str(frame["__group"].dropna().iloc[0])
    return csv_path.stem.split("___")[0]


def infer_run_id(csv_path: Path, frame: pd.DataFrame) -> str:
    if "__run_id" in frame.columns and not frame["__run_id"].dropna().empty:
        return str(frame["__run_id"].dropna().iloc[0])
    return csv_path.stem


def load_plot_frame(root: Path, cfg: DictConfig) -> pd.DataFrame:
    frames = []
    csv_paths = sorted(path for path in root.rglob("*.csv") if is_run_csv(path))
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = [column for column in [cfg.data.x_key, cfg.data.y_key] if column not in frame]
        if missing:
            Logger.detail(f"skipping {csv_path}: missing {missing}", color="yellow")
            continue

        group = infer_group_name(csv_path, frame)
        run_id = infer_run_id(csv_path, frame)
        group = cfg.labels.rename.get(group, group)
        frame = frame[[cfg.data.x_key, cfg.data.y_key]].copy()
        frame[cfg.data.hue_column] = group
        frame["__run_id"] = run_id
        frame["__source_file"] = str(csv_path)
        frames.append(frame)

    if not frames:
        raise ValueError(f"No plottable CSV files found in {root}")
    return pd.concat(frames, ignore_index=True)


def make_alignment_grid(frame: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    align_cfg = cfg.plot.get("align", {})
    x_values = pd.to_numeric(frame[cfg.data.x_key], errors="coerce").dropna()
    if x_values.empty:
        raise ValueError(f"No finite values found for x key '{cfg.data.x_key}'")

    min_x = align_cfg.get("min_x", None)
    max_x = align_cfg.get("max_x", None)
    grid_min = float(min_x) if min_x is not None else float(x_values.min())
    grid_max = float(max_x) if max_x is not None else float(x_values.max())

    if bool(align_cfg.get("overlap_only", False)):
        run_ranges = []
        for _, run_frame in frame.groupby("__run_id", sort=False):
            run_x = pd.to_numeric(run_frame[cfg.data.x_key], errors="coerce").dropna()
            if not run_x.empty:
                run_ranges.append((float(run_x.min()), float(run_x.max())))
        if not run_ranges:
            raise ValueError("Cannot build overlap x-grid because no run has finite x values")
        if min_x is None:
            grid_min = max(start for start, _ in run_ranges)
        if max_x is None:
            grid_max = min(end for _, end in run_ranges)

    if grid_max <= grid_min:
        raise ValueError(f"Invalid alignment grid range: min={grid_min}, max={grid_max}")
    return np.linspace(grid_min, grid_max, int(align_cfg.get("n_points", 200)))


def align_frame_to_grid(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    align_cfg = cfg.plot.get("align", {})
    if not bool(align_cfg.get("enabled", False)):
        return frame

    grid = make_alignment_grid(frame, cfg)
    x_key = cfg.data.x_key
    y_key = cfg.data.y_key
    hue_key = cfg.data.hue_column
    aligned_frames = []

    group_keys = [hue_key, "__run_id", "__source_file"]
    for key_values, run_frame in frame.groupby(group_keys, sort=False, dropna=False):
        hue, run_id, source_file = key_values
        run_frame = run_frame.copy()
        run_frame[x_key] = pd.to_numeric(run_frame[x_key], errors="coerce")
        run_frame[y_key] = pd.to_numeric(run_frame[y_key], errors="coerce")
        run_frame = run_frame.dropna(subset=[x_key, y_key]).sort_values(x_key)
        run_frame = run_frame.drop_duplicates(subset=[x_key], keep="last")
        if len(run_frame) < 2:
            continue

        x_values = run_frame[x_key].to_numpy(dtype=float)
        y_values = run_frame[y_key].to_numpy(dtype=float)
        y_grid = np.interp(grid, x_values, y_values, left=np.nan, right=np.nan)
        aligned = pd.DataFrame({x_key: grid, y_key: y_grid})
        aligned[hue_key] = hue
        aligned["__run_id"] = run_id
        aligned["__source_file"] = source_file
        aligned_frames.append(aligned.dropna(subset=[y_key]))

    if not aligned_frames:
        raise ValueError("Alignment removed all runs; check x/y columns and grid limits")
    aligned = pd.concat(aligned_frames, ignore_index=True)
    Logger.detail(
        f"aligned x-grid: {len(grid)} points from {grid[0]:.4g} to {grid[-1]:.4g}"
    )
    return aligned


def log_monotonic_diagnostics(
    frame: pd.DataFrame,
    cfg: DictConfig,
    label: str,
    by_run: bool,
) -> None:
    if not bool(cfg.plot.get("debug_monotonic", False)):
        return

    x_key = cfg.data.x_key
    y_key = cfg.data.y_key
    hue_key = cfg.data.hue_column
    group_keys = [hue_key, "__run_id"] if by_run else [hue_key]
    violations = []

    for key_values, group_frame in frame.groupby(group_keys, sort=False, dropna=False):
        group_frame = group_frame.sort_values(x_key)
        deltas = pd.to_numeric(group_frame[y_key], errors="coerce").diff()
        bad = deltas < -1e-9
        if bad.any():
            first_index = bad[bad].index[0]
            position = group_frame.index.get_loc(first_index)
            previous_index = group_frame.index[position - 1] if position > 0 else first_index
            violations.append(
                {
                    "key": key_values,
                    "count": int(bad.sum()),
                    "from_x": group_frame.loc[previous_index, x_key],
                    "from_y": group_frame.loc[previous_index, y_key],
                    "to_x": group_frame.loc[first_index, x_key],
                    "to_y": group_frame.loc[first_index, y_key],
                }
            )

    if not violations:
        Logger.detail(f"{label}: no monotonicity violations")
        return

    Logger.item(f"{label}: {len(violations)} decreasing trace(s)", color="yellow")
    for violation in violations[:12]:
        Logger.detail(
            f"{violation['key']}: {violation['count']} decrease(s); "
            f"first {violation['from_x']:.4g}/{violation['from_y']:.4g} -> "
            f"{violation['to_x']:.4g}/{violation['to_y']:.4g}",
            color="yellow",
        )
    if len(violations) > 12:
        Logger.detail(f"... {len(violations) - 12} more", color="yellow")


def aggregate_estimator_frame(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame | None:
    estimator = str(cfg.plot.estimator).lower() if cfg.plot.estimator is not None else ""
    if estimator not in {"mean", "median"}:
        return None

    x_key = cfg.data.x_key
    y_key = cfg.data.y_key
    hue_key = cfg.data.hue_column
    grouped = frame.groupby([hue_key, x_key], as_index=False)[y_key]
    if estimator == "mean":
        return grouped.mean()
    return grouped.median()


def discover_group_names(plot_roots: dict[str, Path]) -> list[str]:
    names = []
    for root in plot_roots.values():
        for csv_path in sorted(path for path in root.rglob("*.csv") if is_run_csv(path)):
            frame = pd.read_csv(csv_path, nrows=5)
            names.append(infer_group_name(csv_path, frame))
    return sorted(dict.fromkeys(names))


def prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def prompt_text(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in {None, ""} else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer if answer else (default or "")


def active_config_path(cfg: DictConfig) -> Path | None:
    try:
        config_name = HydraConfig.get().job.config_name
    except Exception:
        config_name = None
    if not config_name:
        return None
    return Path("configs") / f"{config_name}.yaml"


def save_interactive_config(cfg: DictConfig) -> None:
    Logger.subsection("Save Interactive Choices")
    Logger.detail("Choose how to persist rename/color/order changes.")
    Logger.detail("  o: overwrite active config")
    Logger.detail("  n: create a new config [default]")
    Logger.detail("  i: ignore, use changes only for this run")
    choice = input("Save choice [n/o/i]: ").strip().lower() or "n"
    if choice in {"i", "ignore"}:
        Logger.item("using interactive choices without saving", color="yellow")
        return

    if choice in {"o", "overwrite"}:
        output_path = active_config_path(cfg)
        if output_path is None:
            output_path = Path(prompt_text("Could not infer config path. Save as", "configs/lineplot_custom.yaml"))
    else:
        output_path = Path(
            prompt_text("New config path", str(cfg.interaction.new_config_path))
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_path)
    Logger.item(f"saved config: {output_path}", color="green")


def maybe_configure_labels_interactively(
    cfg: DictConfig, raw_group_names: list[str]
) -> DictConfig:
    if not cfg.get("interaction", {}).get("enabled", True):
        return cfg
    if not sys.stdin.isatty():
        Logger.item("interactive label setup skipped because stdin is not a TTY", color="yellow")
        return cfg
    OmegaConf.set_struct(cfg, False)

    Logger.subsection("Interactive Label Setup")
    Logger.detail("Detected run groups:")
    for name in raw_group_names:
        Logger.detail(f"- {name}")

    changed = False
    rename = dict(cfg.labels.rename)
    colors = dict(cfg.labels.colors)
    order = list(cfg.labels.order)

    if prompt_yes_no("Rename run groups interactively?", default=False):
        for raw_name in raw_group_names:
            current = rename.get(raw_name, raw_name)
            renamed = prompt_text(f"Display name for {raw_name}", current)
            if renamed != raw_name:
                rename[raw_name] = renamed
                changed = True
        cfg.labels.rename = rename

    display_names = sorted(
        dict.fromkeys(rename.get(name, name) for name in raw_group_names)
    )

    if prompt_yes_no("Choose custom colors interactively?", default=False):
        for display_name in display_names:
            current = colors.get(display_name, "")
            color = prompt_text(f"Color for {display_name}", current)
            if color:
                colors[display_name] = color
                changed = True
        cfg.labels.colors = colors

    if prompt_yes_no("Set legend order interactively?", default=False):
        Logger.detail("Current labels:")
        for index, display_name in enumerate(display_names, start=1):
            Logger.detail(f"{index}. {display_name}")
        raw_order = prompt_text("Enter comma-separated order by number", "")
        if raw_order:
            indices = [int(token.strip()) for token in raw_order.split(",") if token.strip()]
            order = [display_names[index - 1] for index in indices]
            order.extend(name for name in display_names if name not in order)
            cfg.labels.order = order
            changed = True

    if changed:
        save_interactive_config(cfg)
    else:
        Logger.item("no label changes requested", color="yellow")
    return cfg


def scale_formatter(scale: float | None):
    if scale is None:
        return None

    def formatter(value, _):
        scaled = value / scale
        if abs(scaled - round(scaled)) < 1e-8:
            return f"{int(round(scaled))}"
        return f"{scaled:.1f}"

    return ticker.FuncFormatter(formatter)


def label_with_scale(label: str, scale: float | None) -> str:
    if scale is None:
        return label
    exponent = int(round(math.log10(scale))) if scale > 0 else 0
    return f"{label} ($\\times 10^{exponent}$)"


def make_errorbar(cfg: DictConfig) -> Any:
    kind = cfg.plot.errorbar.kind
    if kind is None or str(kind).lower() == "none":
        return None
    if kind in {"ci", "pi"}:
        return (kind, cfg.plot.errorbar.level)
    return kind


def ordered_hue_values(frame: pd.DataFrame, cfg: DictConfig) -> list[str]:
    values = list(dict.fromkeys(frame[cfg.data.hue_column].astype(str)))
    configured = [value for value in cfg.labels.order if value in values]
    remaining = sorted(value for value in values if value not in configured)
    return configured + remaining


def global_hue_order(frames_by_title: dict[str, pd.DataFrame], cfg: DictConfig) -> list[str]:
    values = []
    for frame in frames_by_title.values():
        values.extend(frame[cfg.data.hue_column].astype(str).tolist())
    values = list(dict.fromkeys(values))
    configured = [value for value in cfg.labels.order if value in values]
    remaining = sorted(value for value in values if value not in configured)
    return configured + remaining


def layout_for(n_plots: int, cfg: DictConfig) -> tuple[int, int]:
    if cfg.layout.n_rows is not None and cfg.layout.n_cols is not None:
        return int(cfg.layout.n_rows), int(cfg.layout.n_cols)
    if n_plots <= 3:
        return 1, n_plots
    n_cols = int(cfg.layout.max_cols)
    n_rows = math.ceil(n_plots / n_cols)
    return n_rows, n_cols


def apply_axis_config(ax, cfg: DictConfig, is_bottom: bool, is_left: bool) -> None:
    if cfg.axis.log_x:
        ax.set_xscale("log")
    if cfg.axis.log_y:
        ax.set_yscale("log")

    if cfg.axis.x_min is not None or cfg.axis.x_max is not None:
        ax.set_xlim(left=cfg.axis.x_min, right=cfg.axis.x_max)
    if cfg.axis.y_min is not None or cfg.axis.y_max is not None:
        ax.set_ylim(bottom=cfg.axis.y_min, top=cfg.axis.y_max)

    if cfg.axis.x_scale is not None and not cfg.axis.log_x:
        ax.xaxis.set_major_formatter(scale_formatter(float(cfg.axis.x_scale)))
    if cfg.axis.y_scale is not None and not cfg.axis.log_y:
        ax.yaxis.set_major_formatter(scale_formatter(float(cfg.axis.y_scale)))

    ax.set_xlabel(label_with_scale(cfg.axis.x_label, cfg.axis.x_scale) if is_bottom else "")
    ax.set_ylabel(label_with_scale(cfg.axis.y_label, cfg.axis.y_scale) if is_left else "")


def palette_for(hue_values: list[str], cfg: DictConfig) -> dict[str, Any]:
    palette = dict(cfg.labels.colors)
    missing = [value for value in hue_values if value not in palette]
    generated = sns.color_palette(cfg.style.palette, n_colors=len(missing))
    palette.update({value: color for value, color in zip(missing, generated)})
    return palette


def plot_one(
    ax,
    frame: pd.DataFrame,
    title: str,
    cfg: DictConfig,
    hue_order: list[str],
    palette: dict[str, Any],
) -> None:
    sns.lineplot(
        data=frame,
        x=cfg.data.x_key,
        y=cfg.data.y_key,
        hue=cfg.data.hue_column,
        hue_order=hue_order,
        palette=palette,
        estimator=cfg.plot.estimator,
        errorbar=make_errorbar(cfg),
        n_boot=cfg.plot.n_boot,
        linewidth=cfg.plot.linewidth,
        alpha=cfg.plot.line_alpha,
        ax=ax,
    )
    ax.set_title(title)
    ax.grid(True, alpha=cfg.style.grid_alpha)


def configure_style(cfg: DictConfig) -> None:
    preset_rc = {}
    if cfg.style.preset in {"neurips", "icml"}:
        preset_rc = {
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
        }

    sns.set_theme(
        context=cfg.style.context,
        style=cfg.style.style,
        font=cfg.style.font,
        font_scale=cfg.style.font_scale,
        palette=cfg.style.palette,
    )
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": cfg.output.dpi,
            "savefig.dpi": cfg.output.dpi,
            **preset_rc,
        }
    )


def collect_shared_legend(axes, cfg: DictConfig) -> tuple[list[Any], list[str]]:
    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                handles_by_label.setdefault(label, handle)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    labels = [label for label in cfg.labels.order if label in handles_by_label]
    labels += sorted(label for label in handles_by_label if label not in labels)
    return [handles_by_label[label] for label in labels], labels


def make_lineplot(cfg: DictConfig) -> None:
    input_dir = Path(cfg.data.input_dir)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Logger.section("Lineplot")
    Logger.item(f"input: {input_dir}")
    Logger.item(f"output: {output_dir}")
    Logger.item(f"x: {cfg.data.x_key}")
    Logger.item(f"y: {cfg.data.y_key}")

    plot_roots = discover_plot_roots(
        input_dir,
        cfg.data.subplot_mode,
        dict(cfg.labels.titles),
        list(cfg.labels.get("panel_order", [])),
    )
    Logger.item(f"panels: {len(plot_roots)}", color="green")

    raw_group_names = discover_group_names(plot_roots)
    cfg = maybe_configure_labels_interactively(cfg, raw_group_names)

    configure_style(cfg)
    frames_by_title = {}
    for title, root in plot_roots.items():
        frame = load_plot_frame(root, cfg)
        log_monotonic_diagnostics(frame, cfg, f"{title} raw runs", by_run=True)
        frame = align_frame_to_grid(frame, cfg)
        log_monotonic_diagnostics(frame, cfg, f"{title} aligned runs", by_run=True)
        estimator_frame = aggregate_estimator_frame(frame, cfg)
        if estimator_frame is not None:
            log_monotonic_diagnostics(
                estimator_frame,
                cfg,
                f"{title} {cfg.plot.estimator} line",
                by_run=False,
            )
        frames_by_title[title] = frame
        Logger.detail(
            f"{title}: {frame['__run_id'].nunique()} runs, "
            f"{frame[cfg.data.hue_column].nunique()} groups"
        )

    hue_order = global_hue_order(frames_by_title, cfg)
    palette = palette_for(hue_order, cfg)
    Logger.detail("legend order: " + ", ".join(hue_order))

    n_plots = len(frames_by_title)
    n_rows, n_cols = layout_for(n_plots, cfg)
    width = cfg.layout.width_per_col * n_cols
    height = cfg.layout.height_per_row * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)
    flat_axes = list(axes.flatten())

    for index, (title, frame) in enumerate(frames_by_title.items()):
        ax = flat_axes[index]
        plot_one(ax, frame, title, cfg, hue_order, palette)
        row = index // n_cols
        col = index % n_cols
        apply_axis_config(
            ax,
            cfg,
            is_bottom=row == n_rows - 1 or index + n_cols >= n_plots,
            is_left=col == 0,
        )

    for ax in flat_axes[n_plots:]:
        ax.set_visible(False)

    if cfg.plot.title:
        fig.suptitle(cfg.plot.title, y=0.995)

    if n_plots > 1 or cfg.legend.location == "bottom":
        handles, labels = collect_shared_legend(flat_axes[:n_plots], cfg)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(labels), int(cfg.legend.max_columns)),
            frameon=cfg.legend.frameon,
        )
        bottom = cfg.legend.bottom_margin
    else:
        ax = flat_axes[0]
        handles, labels = ax.get_legend_handles_labels()
        labels_ordered = [label for label in cfg.labels.order if label in labels]
        labels_ordered += [label for label in labels if label not in labels_ordered]
        handle_lookup = dict(zip(labels, handles))
        ax.legend(
            [handle_lookup[label] for label in labels_ordered],
            labels_ordered,
            loc=cfg.legend.single_location,
            frameon=cfg.legend.frameon,
        )
        bottom = 0.08

    fig.tight_layout(rect=[0, bottom, 1, 0.96 if cfg.plot.title else 1])

    for extension in cfg.output.formats:
        output_path = output_dir / f"{cfg.output.name}.{extension}"
        fig.savefig(output_path, bbox_inches="tight")
        Logger.item(f"saved: {output_path}", color="green")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="lineplot")
def main(cfg: DictConfig) -> None:
    Logger.section("Configuration")
    Logger.detail(OmegaConf.to_yaml(cfg).rstrip())
    make_lineplot(cfg)


if __name__ == "__main__":
    main()
