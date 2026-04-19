from typing import Any


def get_env_id(reference: Any) -> str:
    if isinstance(reference, str):
        return reference

    env = getattr(reference, "unwrapped", reference)
    spec = getattr(env, "spec", None)
    env_id = getattr(spec, "id", None)
    if env_id is not None:
        return env_id
    return env.__class__.__name__


def get_env_module(reference: Any) -> str:
    if isinstance(reference, str):
        return ""

    env = getattr(reference, "unwrapped", reference)
    return getattr(env.__class__, "__module__", "")


def coerce_dict(value: Any, key_name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{key_name}' to be a dict, got {type(value).__name__}")
    return dict(value)
