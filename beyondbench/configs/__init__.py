"""
BeyondBench configuration loading and validation.

Priority order (highest to lowest):
  1. CLI arguments (handled in main.py)
  2. Environment variables (BEYONDBENCH_*)
  3. ``.env`` file in the current working directory (optional — requires
     ``python-dotenv``; values are loaded into ``os.environ`` so they feed
     the standard env-var path without special-casing)
  4. Config file values
  5. Schema defaults
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.json"

# Track whether we've already attempted a .env load this process, so repeated
# load_config() calls don't spam the logs or re-read the file.
_DOTENV_LOADED: bool = False

# Mapping from environment variable name → (nested key path, type)
_ENV_VARS: Dict[str, tuple] = {
    "BEYONDBENCH_MODEL_ID":            (("model", "model_id"),          str),
    "BEYONDBENCH_BACKEND":             (("model", "backend"),            str),
    "BEYONDBENCH_API_KEY":             (("model", "api_key"),            str),
    "BEYONDBENCH_API_PROVIDER":        (("model", "api_provider"),       str),
    "BEYONDBENCH_CUDA_DEVICE":         (("model", "cuda_device"),        str),
    "BEYONDBENCH_GPU_MEMORY_UTIL":     (("model", "gpu_memory_utilization"), float),
    "BEYONDBENCH_TENSOR_PARALLEL":     (("model", "tensor_parallel_size"), int),
    "BEYONDBENCH_OUTPUT_DIR":          (("output", "output_dir"),        str),
    "BEYONDBENCH_LOG_LEVEL":           (("output", "log_level"),         str),
    "BEYONDBENCH_DATAPOINTS":          (("evaluation", "datapoints"),    int),
    "BEYONDBENCH_SUITE":               (("evaluation", "suite"),         str),
    "BEYONDBENCH_SEED":                (("evaluation", "seed"),          int),
    "BEYONDBENCH_TEMPERATURE":         (("evaluation", "temperature"),   float),
    "BEYONDBENCH_MAX_TOKENS":          (("evaluation", "max_tokens"),    int),
}


def _maybe_load_dotenv(
    search_paths: Optional[Iterable[Path]] = None,
    *,
    force_reload: bool = False,
) -> bool:
    """Load variables from the first discoverable ``.env`` into ``os.environ``.

    No-op when ``python-dotenv`` is not installed or no ``.env`` file exists.
    Returns True when a file was loaded. Idempotent within a process unless
    ``force_reload=True``.

    Search order (first hit wins):
      1. Explicit ``search_paths`` argument, if provided.
      2. ``$BEYONDBENCH_DOTENV`` — absolute/relative file path.
      3. ``./.env`` in the current working directory.
      4. Walk up from CWD until a ``.env`` or the filesystem root is found.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED and not force_reload:
        return False

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        _DOTENV_LOADED = True  # mark attempted so we don't retry every call
        return False

    def _candidates() -> Iterable[Path]:
        if search_paths:
            yield from (Path(p) for p in search_paths)
            return
        explicit = os.environ.get("BEYONDBENCH_DOTENV")
        if explicit:
            yield Path(explicit)
        yield Path.cwd() / ".env"
        # Walk parents up to filesystem root
        parent = Path.cwd()
        for _ in range(8):  # safety cap: don't climb forever
            parent = parent.parent
            yield parent / ".env"
            if parent == parent.parent:
                break

    for candidate in _candidates():
        try:
            if candidate.is_file():
                # override=False → existing real env vars take priority over .env
                load_dotenv(candidate, override=False)
                logger.debug("Loaded .env from %s", candidate)
                _DOTENV_LOADED = True
                return True
        except OSError:
            continue

    _DOTENV_LOADED = True
    return False


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML or JSON config file and apply env var overrides.

    Returns a validated config dict with env vars merged in. Any ``.env``
    file in the current working directory (or one of its parents) is loaded
    into ``os.environ`` first — see :func:`_maybe_load_dotenv` for the full
    search order.
    """
    _maybe_load_dotenv()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            config: Dict[str, Any] = yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix} (use .yaml or .json)")

    # Apply environment variable overrides (env > file)
    config = _apply_env_vars(config)

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate a config dict against the JSON schema.

    Returns True if valid.  Raises jsonschema.ValidationError on failure.
    """
    try:
        import jsonschema
    except ImportError:
        # jsonschema not installed — skip silently
        return True

    if not _SCHEMA_PATH.exists():
        return True

    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    jsonschema.validate(config, schema)
    return True


def load_and_validate_config(path: str | Path) -> Dict[str, Any]:
    """Load config from file, apply env vars, and validate against schema."""
    config = load_config(path)
    validate_config(config)
    return config


def get_env_overrides() -> Dict[str, Any]:
    """Return a nested dict of all BEYONDBENCH_* env vars that are currently set.

    Loads ``.env`` (if present and ``python-dotenv`` is installed) before
    reading the environment, so callers that skip :func:`load_config` still
    see dotenv values.
    """
    _maybe_load_dotenv()
    result: Dict[str, Any] = {}
    for env_var, (key_path, cast) in _ENV_VARS.items():
        raw = os.environ.get(env_var)
        if raw is not None:
            try:
                value = cast(raw)
            except (ValueError, TypeError):
                continue
            _set_nested(result, key_path, value)
    return result


def _apply_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge env var values into config dict (env vars take priority over file)."""
    overrides = get_env_overrides()
    if not overrides:
        return config

    import copy
    config = copy.deepcopy(config)
    _deep_merge(config, overrides)
    return config


def _set_nested(d: Dict[str, Any], key_path: tuple, value: Any) -> None:
    """Set a value in a nested dict given a tuple key path."""
    for key in key_path[:-1]:
        d = d.setdefault(key, {})
    d[key_path[-1]] = value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override into base in-place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def list_presets() -> Dict[str, Path]:
    """Return a mapping of preset name → path for all built-in presets."""
    configs_dir = Path(__file__).parent
    presets = {}
    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        presets[yaml_file.stem] = yaml_file
    return presets


# Re-export ConfigLoader as a class-style façade (Phase 9 API shape).
# The class lives in a sibling module to keep this file compact and to let
# consumers do `from beyondbench.configs.config_loader import ConfigLoader`.
from .config_loader import ConfigLoader  # noqa: E402

# Re-export the model validator so `from beyondbench.configs import ModelValidator`
# works without having to reach into a sub-module.
from .model_validator import (  # noqa: E402
    ModelValidator,
    ValidationIssue,
    ValidationResult,
    validate_model_config,
)


__all__ = [
    "load_config",
    "validate_config",
    "load_and_validate_config",
    "get_env_overrides",
    "list_presets",
    "ConfigLoader",
    "ModelValidator",
    "ValidationIssue",
    "ValidationResult",
    "validate_model_config",
]
