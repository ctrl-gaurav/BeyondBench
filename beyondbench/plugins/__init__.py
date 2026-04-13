"""
BeyondBench Plugin System
=========================

This module provides the plugin infrastructure for extending BeyondBench
with custom task implementations.

Quick start
-----------
Install a package that declares a ``beyondbench.tasks`` entry point,
or point :class:`PluginManager` at a local directory:

.. code-block:: python

    from beyondbench.plugins import PluginManager

    pm = PluginManager(plugin_dir="/path/to/my/plugins")
    pm.load_all_plugins()

Public API
----------
- :class:`PluginManager` — discovers, validates, and registers plugins
- :class:`~beyondbench.plugins.metadata.TaskMetadata` — structured task metadata
- :func:`~beyondbench.plugins.metadata.get_task_metadata` — retrieve metadata for any task class
- :data:`ENTRY_POINT_GROUP` — the setuptools entry-point group name
  (``"beyondbench.tasks"``)
"""

from .discovery import (
    ENTRY_POINT_GROUP,
    PluginManager,
    PluginValidationError,
    validate_plugin_class,
)
from .metadata import (
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
    TaskMetadata,
    get_task_metadata,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "PluginManager",
    "PluginValidationError",
    "validate_plugin_class",
    "VALID_CATEGORIES",
    "VALID_DIFFICULTIES",
    "TaskMetadata",
    "get_task_metadata",
]
