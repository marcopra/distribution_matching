import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing

from env.domain_utils import coerce_dict, get_env_id


LEGACY_ATARI_ID_MARKERS = (
    "NoFrameskip",
    "Deterministic",
)


def _has_atari_entry_point(env_id):
    try:
        spec = gym.spec(env_id)
    except gym.error.Error:
        return False

    entry_point = str(getattr(spec, "entry_point", ""))
    return getattr(spec, "namespace", None) == "ALE" or "ale_py.env" in entry_point


class AtariScoreMaskWrapper(gym.ObservationWrapper):
    """
    Mask Atari score area by overwriting a top band.
    Defaults are tuned per game, but can be overridden in config.
    """

    DEFAULT_BANDS = {
        "ALE/Pong-v5": 10,
        "PongNoFrameskip-v4": 10,
        "ALE/Breakout-v5": 12,
        "BreakoutNoFrameskip-v4": 12,
        "ALE/SpaceInvaders-v5": 12,
        "SpaceInvadersNoFrameskip-v4": 12,
        "TennisNoFrameskip-v4": 8,
        "BowlingNoFrameskip-v4": 25,
        "MarioBrosNoFrameskip-v4": 7,
        "ALE/MarioBros-v5": 7,
        "ALE/MontezumaRevenge-v5": 0,
    }

    def __init__(self, env, band_height=None, color=255):
        super().__init__(env)
        self.band_height = band_height
        self.color = color

    def _resolve_band_height(self):
        if self.band_height is not None:
            return self.band_height

        env_name = get_env_id(self.env.unwrapped)
        if env_name in self.DEFAULT_BANDS:
            return self.DEFAULT_BANDS[env_name]
        return 0

    def observation(self, obs):
        if not isinstance(obs, np.ndarray) or obs.ndim != 3:
            return obs

        band = self._resolve_band_height()
        if band <= 0:
            return obs

        out = obs.copy()
        out[:band, :, :] = self.color
        return out


def is_atari_env(reference):
    if not isinstance(reference, str):
        env = getattr(reference, "unwrapped", reference)
        if isinstance(env, ale_py.env.AtariEnv):
            return True

    env_id = get_env_id(reference)
    if env_id.startswith("ALE/") or _has_atari_entry_point(env_id):
        return True

    return any(marker in env_id for marker in LEGACY_ATARI_ID_MARKERS)


def pop_atari_kwargs(env_kwargs):
    atari_kwargs = coerce_dict(env_kwargs.pop("atari", {}), "atari")

    if any(key in env_kwargs for key in ("score_mask", "score_mask_band", "score_mask_color")):
        score_mask_cfg = coerce_dict(atari_kwargs.get("score_mask", {}), "atari.score_mask")
        if "score_mask" in env_kwargs:
            score_mask_cfg.setdefault("enabled", env_kwargs.pop("score_mask"))
        if "score_mask_band" in env_kwargs:
            score_mask_cfg.setdefault("band_height", env_kwargs.pop("score_mask_band"))
        if "score_mask_color" in env_kwargs:
            score_mask_cfg.setdefault("color", env_kwargs.pop("score_mask_color"))
        atari_kwargs["score_mask"] = score_mask_cfg

    return atari_kwargs


def wrap_atari_pixels(env, name, action_repeat, grayscale, atari_kwargs):
    atari_kwargs = coerce_dict(atari_kwargs, "atari")
    score_mask_cfg = coerce_dict(atari_kwargs.pop("score_mask", {}), "atari.score_mask")

    score_mask_enabled = bool(score_mask_cfg.pop("enabled", False))
    score_mask_band = score_mask_cfg.pop("band_height", None)
    score_mask_color = score_mask_cfg.pop("color", 255)
    if score_mask_cfg:
        unknown_keys = ", ".join(sorted(score_mask_cfg))
        raise TypeError(f"Unknown Atari score mask kwargs: {unknown_keys}")

    preprocessing_kwargs = {
        "noop_max": 0,
        "frame_skip": action_repeat,
        "screen_size": 84,
        "terminal_on_life_loss": False,
        "grayscale_obs": grayscale,
        "grayscale_newaxis": grayscale,
        "scale_obs": False,
    }
    preprocessing_kwargs.update(atari_kwargs)
    preprocessing_kwargs["frame_skip"] = action_repeat
    preprocessing_kwargs["grayscale_obs"] = grayscale
    preprocessing_kwargs["grayscale_newaxis"] = grayscale

    env = AtariPreprocessing(env, **preprocessing_kwargs)
    if score_mask_enabled:
        env = AtariScoreMaskWrapper(env, band_height=score_mask_band, color=score_mask_color)

    return env, 1
