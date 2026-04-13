"""
Plugin discovery and loading for BeyondBench.

Supports two discovery mechanisms:

1. **Entry point discovery** — packages that install themselves under the
   ``beyondbench.tasks`` entry-point group are automatically found via
   ``importlib.metadata.entry_points()``.

2. **Local directory discovery** — a directory path can be passed to
   :class:`PluginManager` to load ``.py`` files or packages from disk
   without requiring installation (useful during development).
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# The entry-point group name used by BeyondBench plugins
ENTRY_POINT_GROUP = "beyondbench.tasks"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class PluginValidationError(Exception):
    """Raised when a plugin fails validation."""


def validate_plugin_class(
    task_class: type,
    existing_tasks: Optional[Dict[str, type]] = None,
) -> Tuple[bool, str]:
    """
    Validate that *task_class* is a well-formed BeyondBench task plugin.

    Returns ``(True, "")`` on success or ``(False, reason)`` on failure.

    Checks
    ------
    1. Must be a class.
    2. Must be a subclass of :class:`~beyondbench.core.base_task.BaseTask`.
    3. Must have a ``task_name`` property / attribute that returns a
       non-empty string.
    4. Must implement all abstract methods:
       ``generate_data``, ``create_prompt``, ``evaluate_response``.
    5. Must not conflict with an already-registered task name.
    """
    from ..core.base_task import BaseTask  # local import to avoid circular

    # 1. Must be a class
    if not inspect.isclass(task_class):
        return False, f"Expected a class, got {type(task_class).__name__}"

    # 2. Must subclass BaseTask
    if not issubclass(task_class, BaseTask):
        return False, (
            f"'{task_class.__name__}' must inherit from BaseTask "
            f"(beyondbench.core.base_task.BaseTask)"
        )

    # 3. Must expose task_name
    task_name: str = _resolve_task_name(task_class)
    if not task_name:
        return False, (
            f"'{task_class.__name__}' must have a non-empty 'task_name' "
            f"property or class attribute"
        )

    # 4. Abstract method implementations
    required_methods = ["generate_data", "create_prompt", "evaluate_response"]
    for method_name in required_methods:
        method = getattr(task_class, method_name, None)
        if method is None:
            return False, (
                f"'{task_class.__name__}' is missing required method "
                f"'{method_name}'"
            )
        # Check it's not still abstract
        if getattr(method, "__isabstractmethod__", False):
            return False, (
                f"'{task_class.__name__}.{method_name}' is still abstract — "
                f"you must provide a concrete implementation"
            )

    # 5. Name conflict with built-in tasks
    if existing_tasks and task_name in existing_tasks:
        return False, (
            f"Plugin task name '{task_name}' conflicts with an existing "
            f"built-in task.  Choose a unique name."
        )

    return True, ""


def _resolve_task_name(task_class: type) -> str:
    """Try to read task_name from the class without instantiating it."""
    prop = getattr(task_class, "task_name", None)
    if prop is None:
        return ""
    if isinstance(prop, str):
        return prop.strip()
    if isinstance(prop, property):
        # Try calling the fget on a mock; fall back to deriving from class name
        try:
            return prop.fget(None)  # type: ignore[arg-type]
        except Exception:
            pass
        # Derive from class name
        raw = task_class.__name__.lower()
        for suffix in ("task",):
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)].rstrip("_")
        return raw
    # Descriptor or something else
    return str(prop).strip()


# ---------------------------------------------------------------------------
# Entry-point helpers
# ---------------------------------------------------------------------------

def _iter_entry_points():
    """Yield (name, ep) tuples from the ``beyondbench.tasks`` group.

    Compatible with Python 3.9+ ``importlib.metadata`` API.
    """
    try:
        from importlib.metadata import entry_points as _eps

        # Python >=3.12 / 3.9 with keyword argument
        try:
            eps = _eps(group=ENTRY_POINT_GROUP)
        except TypeError:
            # Python 3.9 compatibility
            all_eps = _eps()
            eps = all_eps.get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

        for ep in eps:
            yield ep.name, ep
    except ImportError:
        # Very old Python — no entry points support
        return


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class PluginManager:
    """
    Discovers, loads, validates, and registers BeyondBench task plugins.

    Usage
    -----
    .. code-block:: python

        from beyondbench.plugins import PluginManager
        from beyondbench.core.task_registry import TaskRegistry

        registry = TaskRegistry.__new__(TaskRegistry)  # bare instance
        pm = PluginManager(registry)
        pm.load_all_plugins()

    Parameters
    ----------
    registry : TaskRegistry, optional
        The registry to register discovered tasks into.  If ``None``, a
        fresh :class:`~beyondbench.core.task_registry.TaskRegistry` is
        created (which already runs built-in discovery).
    plugin_dir : str or Path, optional
        Additional local directory to scan for plugin modules.
    default_suite : str
        Suite to assign plugins that don't declare their own difficulty.
    """

    def __init__(
        self,
        registry=None,
        plugin_dir: Optional[str | Path] = None,
        default_suite: str = "easy",
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        if registry is None:
            from ..core.task_registry import TaskRegistry
            registry = TaskRegistry()
        self.registry = registry
        self.plugin_dir: Optional[Path] = Path(plugin_dir) if plugin_dir else None
        self.default_suite = default_suite

        # Track loaded plugins: name → class
        self._loaded: Dict[str, type] = {}
        # Errors collected during loading
        self._errors: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all_plugins(self) -> Dict[str, type]:
        """
        Run full plugin discovery: entry points + local directory.

        Returns a dict of ``{task_name: task_class}`` for all successfully
        loaded plugins.
        """
        self._load_from_entry_points()
        if self.plugin_dir is not None:
            self._load_from_directory(self.plugin_dir)
        return dict(self._loaded)

    def load_from_entry_points(self) -> Dict[str, type]:
        """Load only entry-point plugins."""
        self._load_from_entry_points()
        return dict(self._loaded)

    def load_from_directory(self, directory: str | Path) -> Dict[str, type]:
        """Load plugins from a local directory (dev-mode)."""
        self._load_from_directory(Path(directory))
        return dict(self._loaded)

    @property
    def loaded_plugins(self) -> Dict[str, type]:
        """Return a snapshot of successfully loaded plugins."""
        return dict(self._loaded)

    @property
    def errors(self) -> List[str]:
        """Return a list of error messages from failed plugin loads."""
        return list(self._errors)

    # ------------------------------------------------------------------
    # Internal discovery
    # ------------------------------------------------------------------

    def _load_from_entry_points(self) -> None:
        self.logger.debug("Scanning '%s' entry points …", ENTRY_POINT_GROUP)
        count = 0
        for ep_name, ep in _iter_entry_points():
            try:
                obj = ep.load()
            except Exception as exc:
                msg = f"Failed to load entry point '{ep_name}': {exc}"
                self.logger.warning(msg)
                self._errors.append(msg)
                continue

            # ep may resolve to a module (plugin exposes multiple tasks) or
            # directly to a class.
            if inspect.isclass(obj):
                self._register_candidate(ep_name, obj)
                count += 1
            elif inspect.ismodule(obj):
                # Scan the module for task classes
                found = self._find_task_classes_in_module(obj)
                for cls in found:
                    self._register_candidate(ep_name, cls)
                    count += 1
            else:
                msg = (
                    f"Entry point '{ep_name}' resolved to unsupported type "
                    f"{type(obj).__name__} (expected a class or module)"
                )
                self.logger.warning(msg)
                self._errors.append(msg)

        self.logger.debug("Entry-point scan complete: %d task(s) loaded.", count)

    def _load_from_directory(self, directory: Path) -> None:
        if not directory.exists():
            self.logger.warning("Plugin directory '%s' does not exist.", directory)
            return
        if not directory.is_dir():
            self.logger.warning("'%s' is not a directory.", directory)
            return

        self.logger.debug("Scanning plugin directory '%s' …", directory)

        # Collect .py files (not __init__ at top level — those are packages)
        candidates: List[Path] = []

        for item in sorted(directory.iterdir()):
            if item.is_file() and item.suffix == ".py" and item.name != "__init__.py":
                candidates.append(item)
            elif item.is_dir() and (item / "__init__.py").exists():
                # It's a package — import via __init__.py
                candidates.append(item / "__init__.py")

        for py_file in candidates:
            module = self._load_module_from_path(py_file)
            if module is None:
                continue
            for cls in self._find_task_classes_in_module(module):
                self._register_candidate(py_file.stem, cls)

    def _load_module_from_path(self, py_path: Path):
        """Dynamically import a .py file and return the module (or None)."""
        module_name = f"_beyondbench_plugin_{py_path.stem}_{abs(hash(str(py_path)))}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_path)
            if spec is None or spec.loader is None:
                self.logger.warning("Could not create spec for '%s'", py_path)
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            return module
        except Exception as exc:
            msg = f"Failed to import plugin from '{py_path}': {exc}"
            self.logger.warning(msg)
            self._errors.append(msg)
            return None

    def _find_task_classes_in_module(self, module) -> List[type]:
        """Return all BaseTask subclasses defined in *module*."""
        from ..core.base_task import BaseTask

        results = []
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseTask)
                and obj is not BaseTask
                and obj.__module__ == module.__name__
            ):
                results.append(obj)
        return results

    def _register_candidate(self, ep_name: str, task_class: type) -> None:
        """Validate and register a single candidate class."""
        ok, reason = validate_plugin_class(
            task_class, existing_tasks=self.registry._tasks
        )
        if not ok:
            msg = (
                f"Plugin '{ep_name}' ({task_class.__name__}) rejected: {reason}"
            )
            self.logger.warning(msg)
            self._errors.append(msg)
            return

        task_name = _resolve_task_name(task_class)

        # Determine suite from metadata if available
        suite = self.default_suite
        from .metadata import get_task_metadata
        try:
            meta = get_task_metadata(task_class, suite=suite)
            suite = meta.difficulty
        except Exception:
            pass

        try:
            self.registry.register_task(task_name, task_class, suite)
            self._loaded[task_name] = task_class
            self.logger.info(
                "Loaded plugin task '%s' (%s) into suite '%s'",
                task_name,
                task_class.__name__,
                suite,
            )
        except Exception as exc:
            msg = f"Failed to register plugin task '{task_name}': {exc}"
            self.logger.warning(msg)
            self._errors.append(msg)
