#!/usr/bin/env python3
"""Collect real PointMaze rollouts and render exploration comparison video.

Examples:
    python make_exploration_video.py --mode all
    python make_exploration_video.py --mode render
    python make_exploration_video.py --mode collect
    python make_exploration_video.py --mode all --episodes 10 --umaze-seed 61200 --largedense-seed 71200

Use the project's ``dist_matching`` conda environment when its dependencies are
not installed in the active interpreter. Collection caches each environment /
method / episode independently; rendering reads cache only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import gc
import numpy as np
from pathlib import Path

# ---------------------------- Frequently edited settings --------------------
ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT / "models" / "pointmaze_new"
OUTPUT_DIR = ROOT / "exploration_video"
CACHE_DIR = OUTPUT_DIR / "trajectory_cache"
VIDEO_PATH = OUTPUT_DIR / "rover_exploration_comparison.mp4"

# Fixed column positions: Random far left; Rover centered among seven methods.
METHODS = [
    {"id": "random", "label": "Random", "kind": "random"},
    {"id": "rnd", "label": "RND", "kind": "snapshot"},
    {"id": "smm", "label": "SMM", "kind": "snapshot"},
    {"id": "rover", "label": "Rover (Ours)", "kind": "snapshot"},
    {"id": "icm_apt", "label": "APT", "kind": "snapshot"},
    {"id": "maxent", "label": "MAXENT", "kind": "snapshot"},
    {"id": "cic", "label": "CIC", "kind": "snapshot"},
]
ENVIRONMENTS = ["umaze", "largedense"]
EVALUATION_SEEDS = {"umaze": 61200, "largedense": 71200}
EPISODES = 10
CHECKPOINT_STEP = 1_000_000
DEVICE = "cuda"  # fallback to CPU if CUDA unavailable
CAMERA_VIEW = "fixed top-down world XY"
CHECKPOINT_RELATIVE_PATHS = {
    "umaze": {
        "rover": "models/pointmaze_new/umaze/states/rover/models/snapshot_1000000.pt",
        "rnd": "models/pointmaze_new/umaze/states/rnd/models/final_snapshot.pt",
        "smm": "models/pointmaze_new/umaze/states/smm/models/final_snapshot.pt",
        "icm_apt": "models/pointmaze_new/umaze/states/icm_apt/models/final_snapshot.pt",
        "maxent": "models/pointmaze_new/umaze/states/maxent/models/final_snapshot.pt",
        "cic": "models/pointmaze_new/umaze/states/cic/models/final_snapshot.pt",
    },
    "largedense": {
        "rover": "models/pointmaze_new/largedense/states/rover/models/snapshot_1000000.pt",
        "rnd": "models/pointmaze_new/largedense/states/rnd/models/states/gym/rnd/0/snapshot_1000000.pt",
        "smm": "models/pointmaze_new/largedense/states/smm/models/states/gym/smm/0/snapshot_1000000.pt",
        "icm_apt": "models/pointmaze_new/largedense/states/icm_apt/models/states/gym/icm_apt/0/snapshot_1000000.pt",
        "maxent": "models/pointmaze_new/largedense/states/maxent/models/states/gym/maxent/0/snapshot_1000000.pt",
        "cic": "models/pointmaze_new/largedense/states/cic/models/states/gym/cic/0/snapshot_1000000.pt",
    },
}

# Render settings. Edit any of these without changing trajectory cache.
EPISODE_SECONDS = 5.0
LABEL_REVEAL_EPISODE = 3
LABEL_FADE_SECONDS = 0.8
FINAL_HOLD_SECONDS = 3.0
FPS = 30
WIDTH, HEIGHT = 1920, 1080
BACKGROUND = "#FFFFFF"
PANEL_BACKGROUND = "#FFFFFF"
FLOOR_COLOR = "#FFFFFF"
WALL_COLOR = "#30415C"
WALL_OUTLINE = "#0A1019"
# Matplotlib plasma samples, matching the per-trajectory maze overlay plots.
TRAJECTORY_PALETTE = (
    "#5C01A6", "#7901A8", "#9613A1", "#B3318D", "#CC4A78",
    "#E16462", "#EF8050", "#F69F3E", "#F6BF33", "#E8DE2A",
)
METHOD_LABEL_COLOR = "#30415C"
ROVER_LABEL_COLOR = "#D63D46"
TRAIL_WIDTH = 2
HISTORY_OPACITY = 0.27
CURRENT_OPACITY = 0.96
AGENT_POINT_RADIUS = 5
FINAL_HISTORY_OPACITY = 1.0
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ENVIRONMENT_LABEL_FONT_SIZE = 34
EPISODE_FONT_SIZE = 20
METHOD_LABEL_FONT_SIZE = 27
SMALL_FONT_SIZE = 16
PANEL_GAP = 12
MAZE_INSET = 2
PANEL_TOP = 82
PANEL_BOTTOM = 467
SECOND_ROW_TOP = 555
SECOND_ROW_BOTTOM = 930
METHOD_LABEL_Y = 947
FOOTER_Y = 1034
# ----------------------------------------------------------------------------

METHOD_ORDER = [m["id"] for m in METHODS]
CACHE_SCHEMA = 1


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_path(environment: str, method: dict) -> Path | None:
    if method["kind"] == "random":
        return None
    configured = CHECKPOINT_RELATIVE_PATHS.get(environment, {}).get(method["id"])
    if configured:
        return ROOT / configured
    root = EXPERIMENT_ROOT / environment / "states" / method["id"] / "models"
    candidates = [
        root / f"snapshot_{CHECKPOINT_STEP}.pt",
        root / f"states/gym/{method['id']}/0/snapshot_{CHECKPOINT_STEP}.pt",
        root / "final_snapshot.pt",
        root / f"pointmaze/{environment}_goal_1/states/baselines/{method['id']}_discrete/seed_1/best_snapshot.pt",
    ]
    return next((p for p in candidates if p.is_file()), None)


def _config_path(environment: str, method: dict) -> Path:
    path = EXPERIMENT_ROOT / environment / "states" / method["id"] / ".hydra" / "config.yaml"
    if path.is_file():
        return path
    fallback = ROOT / "configs/env/pointmaze" / (
        "pointmaze_umaze_goal_1.yaml" if environment == "umaze" else "pointmaze_largedense_goal_1.yaml"
    )
    return fallback


def _file_identity(path: Path | None) -> dict | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _eval_settings(environment: str, method: dict) -> dict:
    checkpoint = _checkpoint_path(environment, method)
    return {
        "schema": CACHE_SCHEMA,
        "environment": environment,
        "method": method["id"],
        "checkpoint": _file_identity(checkpoint),
        "checkpoint_step": 0 if method["id"] == "random" else CHECKPOINT_STEP,
        "seed": EVALUATION_SEEDS[environment],
        "episode_count": EPISODES,
        "horizon": 500 if environment == "umaze" else 2000,
        "config": str(_config_path(environment, method).relative_to(ROOT)),
        "config_sha256": hashlib.sha256(_config_path(environment, method).read_bytes()).hexdigest(),
        "deterministic": False,
        "start_position_variance": 0.0,
        "skill_protocol": "one fixed latent sampled per episode using repository trajectory helper" if method["id"] in {"cic", "smm"} else "agent.init_meta/update_meta",
        "episode_seed_rule": "base_seed + episode_index; same across methods",
    }


def _cache_file(environment: str, method: str, episode: int) -> Path:
    return CACHE_DIR / environment / method / f"episode_{episode:02d}.npz"


def _cache_invalid_reason(path: Path, settings: dict, allow_mixed_seeds: bool = False) -> str | None:
    if not path.is_file():
        return "file is missing"
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            points = data["positions"]
        cached_settings = dict(meta.get("settings", {}))
        requested_settings = dict(settings)
        # Episode count changes collection completeness, not identity of an
        # already collected per-episode rollout. This lets 5 -> 10 add only
        # episodes 6-10 while keeping their exact original cache records.
        cached_settings.pop("episode_count", None)
        requested_settings.pop("episode_count", None)
        # Rendering can combine runs with different evaluation seeds. Keep
        # environment, method, checkpoint, and config checks strict so copied
        # episodes cannot silently land in the wrong maze or policy column.
        if allow_mixed_seeds:
            cached_settings.pop("seed", None)
            requested_settings.pop("seed", None)
        mismatched = [
            key for key in set(cached_settings) | set(requested_settings)
            if cached_settings.get(key) != requested_settings.get(key)
        ]
        if mismatched:
            details = ", ".join(
                f"{key}: file={cached_settings.get(key)!r}, expected={requested_settings.get(key)!r}"
                for key in sorted(mismatched)
            )
            return f"incompatible cache metadata ({details})"
        if points.ndim != 2 or points.shape[1] != 2 or len(points) <= 1:
            return f"invalid positions array shape {points.shape}"
        return None
    except Exception as exc:
        return f"could not read cache ({exc})"


def _cache_is_valid(path: Path, settings: dict, allow_mixed_seeds: bool = False) -> bool:
    return _cache_invalid_reason(path, settings, allow_mixed_seeds) is None


def _get_specs():
    specs = []
    missing = []
    for environment in ENVIRONMENTS:
        for method in METHODS:
            checkpoint = _checkpoint_path(environment, method)
            if method["kind"] != "random" and checkpoint is None:
                missing.append(f"{environment}/{method['id']}: no state checkpoint found under {EXPERIMENT_ROOT / environment / 'states' / method['id'] / 'models'}")
            specs.append((environment, method, checkpoint))
    if missing:
        raise FileNotFoundError("Required evaluation assets missing:\n" + "\n".join(missing))
    return specs


def collect_episode_runs(force_recollect: bool = False) -> None:
    """Run repository policy/eval helpers; cache complete positions per episode."""
    import numpy as np
    import torch
    import utils
    from evaluate_pointmaze_models import _clear_runtime_history
    from plot_pointmaze_snapshot_trajectories import (
        _pointmaze_wall_rectangles,
        _sample_latent_meta_for_plot, _reset_valid_pointmaze_start,
        extract_eval_trajectory_point, load_config, load_snapshot, make_env,
        random_action, snapshot_step, patch_runtime_rover_action_dtype,
    )
    from omegaconf import OmegaConf

    selected_device = torch.device(DEVICE if DEVICE == "cpu" or torch.cuda.is_available() else "cpu")
    manifest = {
        "schema": CACHE_SCHEMA,
        "paper": "2606.21271v1.pdf",
        "methods": METHOD_ORDER,
        "display_names": {m["id"]: m["label"] for m in METHODS},
        "environments": ENVIRONMENTS,
        "evaluation_settings": {},
        "rendering_independent": True,
        "coordinate_convention": CAMERA_VIEW + "; repository PointMaze XY; maze cell rectangles from get_debug_maze_layout",
    }
    geometry_by_env: dict[str, dict] = {}

    for environment, method, checkpoint in _get_specs():
        settings = _eval_settings(environment, method)
        manifest["evaluation_settings"][f"{environment}/{method['id']}"] = settings
        files = [_cache_file(environment, method["id"], i) for i in range(EPISODES)]
        valid = [not force_recollect and _cache_is_valid(p, settings) for p in files]
        if all(valid):
            print(f"cache hit: {environment}/{method['id']} ({EPISODES} episodes)", flush=True)
            continue

        cfg = load_config(_config_path(environment, method))
        if method["kind"] != "random":
            agent, payload = load_snapshot(checkpoint, selected_device)
            step = snapshot_step(checkpoint, payload)
            if step == 0:
                step = CHECKPOINT_STEP
            patch_runtime_rover_action_dtype(agent)
        else:
            agent, step = None, 0
        env = make_env(cfg, seed=EVALUATION_SEEDS[environment])
        try:
            _clear_runtime_history(agent)
            walls = _pointmaze_wall_rectangles(env)
            layout_fn = getattr(env, "get_debug_maze_layout", None)
            layout = layout_fn() if callable(layout_fn) else None
            if layout is None:
                raise RuntimeError(f"{environment}: environment supplied no debug maze geometry")
            geometry_by_env[environment] = {k: np.asarray(v).tolist() for k, v in layout.items()}
            horizon = 500 if environment == "umaze" else 2000
            for episode, cache_path in enumerate(files):
                if valid[episode]:
                    continue
                episode_seed = EVALUATION_SEEDS[environment] + episode
                _seed_everything(episode_seed)
                time_step, rejected = _reset_valid_pointmaze_start(env, episode_seed, walls)
                meta = agent.init_meta() if agent is not None and callable(getattr(agent, "init_meta", None)) else {}
                if agent is not None:
                    meta, fixed_latent = _sample_latent_meta_for_plot(
                        agent, meta, np.random.default_rng(episode_seed + 1000003),
                    )
                else:
                    fixed_latent = False
                positions = []
                point = extract_eval_trajectory_point(env, time_step)
                if point is not None:
                    positions.append(point)
                for t in range(horizon):
                    if agent is None:
                        action = random_action(env.action_space, np.random.default_rng(episode_seed * 100000 + t))
                    else:
                        with torch.no_grad(), utils.eval_mode(agent):
                            action = agent.act(time_step.observation, meta, step, eval_mode=False)
                    time_step = env.step(action)
                    update_meta = getattr(agent, "update_meta", None)
                    if agent is not None and callable(update_meta) and not fixed_latent:
                        meta = update_meta(meta, step, time_step)
                    point = extract_eval_trajectory_point(env, time_step)
                    if point is not None:
                        positions.append(point)
                if len(positions) != horizon + 1:
                    raise RuntimeError(f"{environment}/{method['id']}/episode {episode+1}: recorded {len(positions)} states, expected {horizon+1}")
                metadata = {
                    "settings": settings,
                    "episode_index": episode,
                    "episode_seed": episode_seed,
                    "original_steps": horizon,
                    "position_count": len(positions),
                    "rejected_noisy_starts": int(rejected),
                    "checkpoint_step": step,
                    "episode_boundary": "independent reset; no cross-episode segment",
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, positions=np.asarray(positions, dtype=np.float32), metadata=np.asarray(json.dumps(metadata)))
                print(f"collected {environment}/{method['id']} episode {episode+1}/{EPISODES} ({horizon} steps, reset rejects={rejected})", flush=True)
        finally:
            _clear_runtime_history(agent)
            env.close()
            del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Preserve geometry even if all method data were already cached.
    if len(geometry_by_env) < len(ENVIRONMENTS):
        for environment in ENVIRONMENTS:
            if environment not in geometry_by_env:
                cfg = load_config(_config_path(environment, METHODS[0]))
                env = make_env(cfg, seed=EVALUATION_SEEDS[environment])
                try:
                    layout = env.get_debug_maze_layout()
                    geometry_by_env[environment] = {k: np.asarray(v).tolist() for k, v in layout.items()}
                finally:
                    env.close()
    manifest["geometry"] = geometry_by_env
    manifest["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "metadata.json").write_text(json.dumps(manifest, indent=2))


def _load_cached_data():
    import numpy as np
    manifest_path = CACHE_DIR / "metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing cache metadata: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    trajectories = {}
    missing = []
    for environment in ENVIRONMENTS:
        for method in METHODS:
            key = f"{environment}/{method['id']}"
            settings = manifest.get("evaluation_settings", {}).get(key)
            if not settings:
                missing.append(f"{key}: missing evaluation settings metadata")
                continue
            runs = []
            for episode in range(EPISODES):
                path = _cache_file(environment, method["id"], episode)
                invalid_reason = _cache_invalid_reason(path, settings, allow_mixed_seeds=True)
                if invalid_reason is not None:
                    missing.append(f"{key}/episode_{episode+1}: {invalid_reason} at {path}")
                    continue
                with np.load(path, allow_pickle=False) as data:
                    runs.append(data["positions"].astype(np.float32))
            if len(runs) == EPISODES:
                trajectories[(environment, method["id"])] = runs
    if missing:
        raise FileNotFoundError("Render requires complete valid trajectory cache. Missing data:\n" + "\n".join(missing))
    return manifest, trajectories


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    path = FONT_BOLD_PATH if bold else FONT_PATH
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _blend_hex(foreground: str, background: str, alpha: float) -> str:
    fg, bg = foreground.lstrip("#"), background.lstrip("#")
    f = tuple(int(fg[i:i+2], 16) for i in (0, 2, 4))
    b = tuple(int(bg[i:i+2], 16) for i in (0, 2, 4))
    return "#" + "".join(f"{round(x*alpha+y*(1-alpha)):02X}" for x, y in zip(f, b))


def _trajectory_color(episode_index: int) -> str:
    if EPISODES <= 1:
        return TRAJECTORY_PALETTE[0]
    palette_index = round(episode_index * (len(TRAJECTORY_PALETTE) - 1) / (EPISODES - 1))
    return TRAJECTORY_PALETTE[palette_index]


def _draw_panel_base(image, draw, method: dict, environment: str, geometry: dict, col: int, top: int, bottom: int):
    col_w = (WIDTH - 2 * 48 - (len(METHODS)-1) * PANEL_GAP) / len(METHODS)
    x0 = int(48 + col * (col_w + PANEL_GAP))
    x1 = int(x0 + col_w)
    maze_x0, maze_y0 = x0 + MAZE_INSET, top + 14
    maze_x1, maze_y1 = x1 - MAZE_INSET, bottom - 14
    lower = geometry["maze_lower"]
    upper = geometry["maze_upper"]
    xlo, ylo = lower
    xhi, yhi = upper
    world_w, world_h = xhi-xlo, yhi-ylo
    scale = min((maze_x1-maze_x0)/world_w, (maze_y1-maze_y0)/world_h)
    draw_w, draw_h = world_w*scale, world_h*scale
    ox = maze_x0 + (maze_x1-maze_x0-draw_w)/2
    oy = maze_y0 + (maze_y1-maze_y0-draw_h)/2

    def xy_to_px(point):
        return (int(ox+(float(point[0])-xlo)*scale), int(oy+(yhi-float(point[1]))*scale))

    # Flat top-down view. Draw exact wall cells over a white maze floor.
    for rect in geometry["wall_rectangles"]:
        xa, ya, w, h = rect
        left, top_px = xy_to_px((xa, ya+h))
        right, bottom_px = xy_to_px((xa+w, ya))
        draw.rectangle((left, top_px, right, bottom_px), fill=WALL_COLOR)
    # One clear outer edge, with no distracting per-cell grid.
    upper_left = xy_to_px((xlo, yhi))
    lower_right = xy_to_px((xhi, ylo))
    draw.rectangle((*upper_left, *lower_right), outline=WALL_OUTLINE, width=2)
    return (x0, x1), xy_to_px, scale


def _draw_trail(draw, points, upto: float, xy_to_px, color: str, width: int, alpha: float, highlight: bool = False):
    if len(points) < 2:
        return
    total = len(points)-1
    position = max(0.0, min(total, upto*total))
    whole = int(position)
    coords = [xy_to_px(points[i]) for i in range(whole+1)]
    if whole < total:
        t = position-whole
        a, b = points[whole], points[whole+1]
        interp = a + t*(b-a)
        coords.append(xy_to_px(interp))
    if len(coords) >= 2:
        if highlight:
            halo = _blend_hex(color, FLOOR_COLOR, 0.5)
            draw.line(coords, fill=halo, width=width+2, joint="curve")
            draw.line(coords, fill=color, width=width, joint="curve")
        else:
            draw.line(coords, fill=_blend_hex(color, FLOOR_COLOR, alpha), width=width, joint="curve")


def _draw_agent_point(draw, center):
    x, y = center
    radius = AGENT_POINT_RADIUS
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="#D9363E")


def render_video() -> None:
    from PIL import Image, ImageDraw
    import numpy as np
    import imageio_ffmpeg

    manifest, trajectories = _load_cached_data()
    geometry = manifest["geometry"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    total_frames = int(round((EPISODES*EPISODE_SECONDS + FINAL_HOLD_SECONDS)*FPS))
    process = __import__("subprocess").Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(VIDEO_PATH)],
        stdin=__import__("subprocess").PIPE, stdout=__import__("subprocess").DEVNULL, stderr=__import__("subprocess").PIPE,
    )
    env_font, episode_font = _font(ENVIRONMENT_LABEL_FONT_SIZE, True), _font(EPISODE_FONT_SIZE, True)
    label_font, small_font = _font(METHOD_LABEL_FONT_SIZE, True), _font(SMALL_FONT_SIZE)
    for frame_index in range(total_frames):
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        elapsed = frame_index/FPS
        episode_index = min(int(elapsed//EPISODE_SECONDS), EPISODES-1)
        in_animation = elapsed < EPISODES*EPISODE_SECONDS
        episode_progress = (elapsed % EPISODE_SECONDS)/EPISODE_SECONDS if in_animation else 1.0
        draw.text((48, 31), "UMAZE", font=env_font, fill=WALL_COLOR)
        draw.text((48, 499), "LARGEDENSE", font=env_font, fill=WALL_COLOR)
        draw.rounded_rectangle((WIDTH-284, 29, WIDTH-48, 76), radius=16, fill="#F0F3F7", outline="#D5DCE5", width=1)
        draw.text((WIDTH-264, 39), f"Episode {episode_index+1}/{EPISODES}", font=episode_font, fill=WALL_COLOR)
        for row, env_name, top, bottom in ((0, "umaze", PANEL_TOP, PANEL_BOTTOM), (1, "largedense", SECOND_ROW_TOP, SECOND_ROW_BOTTOM)):
            for col, method in enumerate(METHODS):
                (x0x1, map_xyz, map_scale) = _draw_panel_base(image, draw, method, env_name, geometry[env_name], col, top, bottom)
                x0, x1 = x0x1
                runs = trajectories[(env_name, method["id"])]
                # Completed episodes become muted history. Final hold re-highlights all paths.
                end_episode = episode_index if in_animation else EPISODES
                for ep in range(end_episode):
                    color = _trajectory_color(ep)
                    if in_animation:
                        _draw_trail(draw, runs[ep], 1.0, map_xyz, color, TRAIL_WIDTH, HISTORY_OPACITY)
                    else:
                        _draw_trail(draw, runs[ep], 1.0, map_xyz, color, TRAIL_WIDTH, FINAL_HISTORY_OPACITY, highlight=True)
                if in_animation:
                    current_ep = episode_index
                    progress = episode_progress
                    points = runs[current_ep]
                    _draw_trail(draw, points, progress, map_xyz, _trajectory_color(current_ep), TRAIL_WIDTH, CURRENT_OPACITY)
                    idx = min(len(points)-1, int(progress*(len(points)-1)))
                    frac = progress*(len(points)-1)-idx
                    if idx < len(points)-1:
                        pos = points[idx]*(1-frac) + points[idx+1]*frac
                    else:
                        pos = points[-1]
                    _draw_agent_point(draw, map_xyz(pos))
                # Label room is reserved in all frames, names dissolve from episode 3.
                if in_animation and episode_index+1 < LABEL_REVEAL_EPISODE:
                    alpha = 0
                elif in_animation and episode_index+1 == LABEL_REVEAL_EPISODE:
                    alpha = int(255*max(0.0, min(1.0, (elapsed-2*EPISODE_SECONDS)/LABEL_FADE_SECONDS)))
                else:
                    alpha = 255
                if row == 1:
                    label = method["label"]
                    tw = draw.textbbox((0,0), label, font=label_font)[2]
                    if alpha > 0:
                        target_color = ROVER_LABEL_COLOR if method["id"] == "rover" else METHOD_LABEL_COLOR
                        label_color = _blend_hex(target_color, BACKGROUND, alpha / 255.0)
                        draw.text((x0+(x1-x0-tw)//2, METHOD_LABEL_Y), label, font=label_font, fill=label_color)
        draw.text((WIDTH-330, FOOTER_Y), "Playback normalized by episode duration", font=small_font, fill="#748196")
        process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        if frame_index % FPS == 0:
            print(f"render {frame_index}/{total_frames} frames", flush=True)
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", "replace")
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed ({code}): {stderr[-3000:]}")
    print(f"video written: {VIDEO_PATH} ({total_frames} frames)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "render", "all"), default="all")
    parser.add_argument("--episodes", type=int, default=None, help="episodes to collect/render; overrides EPISODES at top")
    parser.add_argument("--umaze-seed", type=int, default=None, help="base evaluation seed for UMAZE")
    parser.add_argument("--largedense-seed", type=int, default=None, help="base evaluation seed for LargeDense")
    parser.add_argument("--force-recollect", action="store_true", help="regenerate all rollout caches; only applies to collect/all")
    args = parser.parse_args()
    global EPISODES
    if args.episodes is not None:
        if args.episodes < 1:
            parser.error("--episodes must be at least 1")
        EPISODES = args.episodes
    if args.umaze_seed is not None:
        EVALUATION_SEEDS["umaze"] = args.umaze_seed
    if args.largedense_seed is not None:
        EVALUATION_SEEDS["largedense"] = args.largedense_seed
    if args.mode == "collect":
        collect_episode_runs(force_recollect=args.force_recollect)
    elif args.mode == "render":
        render_video()
    else:
        collect_episode_runs(force_recollect=args.force_recollect)
        render_video()


if __name__ == "__main__":
    main()
