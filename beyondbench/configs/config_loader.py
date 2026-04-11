"""
Class-style façade over the functional config API (Phase 9 API shape).

The module functions in :mod:`beyondbench.configs` return plain dicts, which
is convenient but a bit awkward to consume with attribute access. The
:class:`ConfigLoader` class below wraps those functions and returns a
:class:`ConfigProxy` with dotted-attribute and dict-style access, plus a
handful of shortcut properties (``cfg.model_name``, ``cfg.backend``, ...).

This keeps two call styles available:

.. code-block:: python

    # Functional — returns a dict
    from beyondbench.configs import load_config, validate_config
    cfg = load_config("beyondbench/configs/default.yaml")
    model_id = cfg["model"]["model_id"]

    # Class-style — returns a ConfigProxy with attribute access
    from beyondbench.configs.config_loader import ConfigLoader
    cfg = ConfigLoader.load("beyondbench/configs/default.yaml")
    model_id = cfg.model_name      # shortcut property
    model_id = cfg.model.model_id  # nested attribute access
    raw_dict = cfg.as_dict()       # escape hatch for code that wants a dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from . import (
    get_env_overrides as _get_env_overrides,
    list_presets as _list_presets,
    load_and_validate_config as _load_and_validate_config,
    load_config as _load_config,
    validate_config as _validate_config,
)


# ---------------------------------------------------------------------------
# ConfigProxy — attribute-and-dict hybrid access over a nested dict
# ---------------------------------------------------------------------------


# Top-level-only shortcut aliases: name → (section, key).
# These only resolve when the proxy wraps the root config dict; on a nested
# proxy they fall through so ``cfg.model.model_id`` returns the actual
# ``model_id`` value instead of being shadowed by the root shortcut.
_ROOT_SHORTCUTS: Dict[str, tuple] = {
    "model_name":   ("model", "model_id"),
    "api_provider": ("model", "api_provider"),
    "api_key":      ("model", "api_key"),
    "cuda_device":  ("model", "cuda_device"),
    "datapoints":   ("evaluation", "datapoints"),
    "suite":        ("evaluation", "suite"),
    "temperature":  ("evaluation", "temperature"),
    "max_tokens":   ("evaluation", "max_tokens"),
    "seed":         ("evaluation", "seed"),
    "top_p":        ("evaluation", "top_p"),
    "output_dir":   ("output", "output_dir"),
    "log_level":    ("output", "log_level"),
}

# These shortcuts fall back to the root config when a direct key on the
# current proxy isn't found. They intentionally overlap with real nested
# keys (e.g. ``cfg.model_id`` on the root proxy returns ``model.model_id``;
# on the nested ``cfg.model`` proxy it returns ``model_id`` directly).
_ROOT_SHORTCUTS_WITH_FALLBACK: Dict[str, tuple] = {
    "model_id": ("model", "model_id"),
    "backend":  ("model", "backend"),
}


class ConfigProxy:
    """Attribute + dict access wrapper around a nested config dict.

    Nested dicts are wrapped on demand, so ``cfg.model.model_id`` returns the
    same value as ``cfg["model"]["model_id"]``. Missing keys raise
    :class:`AttributeError` on attribute access and :class:`KeyError` on
    subscript access, matching normal Python semantics.

    A handful of top-level shortcut aliases provide a flat view of the most-
    used fields — see :data:`_ROOT_SHORTCUTS`. The shortcuts only resolve on
    the root proxy so they never shadow a legitimate nested key
    (``root.model_id`` → ``model.model_id``; ``root.model.model_id`` →
    ``model.model_id`` directly).
    """

    __slots__ = ("_data", "_is_root")

    def __init__(self, data: Dict[str, Any], *, _is_root: bool = True):
        if not isinstance(data, dict):
            raise TypeError(f"ConfigProxy expected dict, got {type(data).__name__}")
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_is_root", _is_root)

    # ---- attribute access ------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Note: __getattr__ only fires when normal attribute lookup fails,
        # so we can freely read _data / _is_root via object.__getattribute__.
        data = object.__getattribute__(self, "_data")
        is_root = object.__getattribute__(self, "_is_root")

        # 1. Direct nested key access
        if name in data:
            value = data[name]
            return ConfigProxy(value, _is_root=False) if isinstance(value, dict) else value

        # 2. Root-only shortcut aliases
        if is_root:
            if name in _ROOT_SHORTCUTS:
                section, key = _ROOT_SHORTCUTS[name]
                return (data.get(section) or {}).get(key)
            if name in _ROOT_SHORTCUTS_WITH_FALLBACK:
                section, key = _ROOT_SHORTCUTS_WITH_FALLBACK[name]
                return (data.get(section) or {}).get(key)

        raise AttributeError(
            f"ConfigProxy has no field '{name}'. Available: {sorted(data.keys())}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_data", "_is_root"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    # ---- dict access -----------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> Any:  # dict-compat
        return self._data.keys()

    def values(self) -> Any:
        return self._data.values()

    def items(self) -> Any:
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """Return the underlying raw dict (not a copy)."""
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the underlying config dict."""
        import copy
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"ConfigProxy({self._data!r})"


# ---------------------------------------------------------------------------
# ConfigLoader — static API that mirrors the verification prompt shape
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Static façade over the functional config API.

    All methods are ``@staticmethod`` so the class acts as a namespace rather
    than something that needs instantiation. Internally it delegates to the
    module functions in :mod:`beyondbench.configs`, so the behavior (schema
    validation, env-var merge, ``.env`` discovery) is identical.
    """

    @staticmethod
    def load(path: Union[str, Path]) -> ConfigProxy:
        """Load a YAML/JSON config, apply env overrides, return a ``ConfigProxy``.

        Raises :class:`FileNotFoundError` when ``path`` does not exist.
        """
        return ConfigProxy(_load_config(path))

    @staticmethod
    def load_and_validate(path: Union[str, Path]) -> ConfigProxy:
        """Like :meth:`load` but validates against the JSON schema too.

        Raises :class:`jsonschema.ValidationError` for schema violations.
        """
        return ConfigProxy(_load_and_validate_config(path))

    @staticmethod
    def validate(config: Union[Dict[str, Any], "ConfigProxy"]) -> bool:
        """Validate ``config`` against the JSON schema.

        Accepts either a raw dict or a ``ConfigProxy``. Returns ``True`` on
        success; raises :class:`jsonschema.ValidationError` on failure.
        """
        if isinstance(config, ConfigProxy):
            config = config.as_dict()
        return _validate_config(config)

    @staticmethod
    def env_overrides() -> Dict[str, Any]:
        """Return the currently-set BEYONDBENCH_* env var overrides as a dict."""
        return _get_env_overrides()

    @staticmethod
    def list_presets() -> Dict[str, Path]:
        """Return ``{preset_name: path}`` for all built-in preset YAMLs."""
        return _list_presets()

    @staticmethod
    def load_preset(name: str) -> ConfigProxy:
        """Load a built-in preset by name (e.g. ``"quick_test"``).

        Raises :class:`ValueError` when the name is unknown.
        """
        presets = _list_presets()
        if name not in presets:
            available: List[str] = sorted(presets.keys())
            raise ValueError(
                f"Unknown preset '{name}'. Available: {', '.join(available)}"
            )
        return ConfigLoader.load_and_validate(presets[name])


__all__ = ["ConfigLoader", "ConfigProxy"]
