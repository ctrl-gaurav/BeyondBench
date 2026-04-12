"""PromptLibrary: singleton registry mapping (task_name, style) → PromptTemplate."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .template import PromptTemplate


class PromptLibrary:
    """Singleton registry of prompt templates, indexed by task name and style.

    Usage::

        lib = PromptLibrary()
        tmpl = lib.get("sum", "concise")
        prompt = tmpl.render(data="[1, 2, 3]")
    """

    _instance: Optional["PromptLibrary"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> "PromptLibrary":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Guard against repeated __init__ calls on the singleton
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._registry: Dict[str, Dict[str, PromptTemplate]] = {}
            # Eagerly load all task prompts.
            # task_prompts.py writes into its own module-level _registry dict.
            # After the import, we merge that dict into self._registry.
            # NOTE: task_prompts.py must NOT call PromptLibrary() or PromptLibrary.__new__()
            # here (it would deadlock since we hold self._lock).  It only uses
            # PromptTemplate and its own module-level _registry dict.
            from . import task_prompts as tp
            # Merge task_prompts' internal registry into our own
            self._registry.update(tp._registry)
            PromptLibrary._initialized = True

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, task_name: str, style: str, template: PromptTemplate) -> None:
        """Register *template* for (*task_name*, *style*).

        Overwrites any previously registered template for the same pair.
        """
        if task_name not in self._registry:
            self._registry[task_name] = {}
        self._registry[task_name][style] = template

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, task_name: str, style: str = "concise") -> PromptTemplate:
        """Return the PromptTemplate for (*task_name*, *style*).

        Raises
        ------
        KeyError
            If no template is registered for the requested task / style.
        """
        task_styles = self._registry.get(task_name)
        if task_styles is None:
            raise KeyError(f"No prompts registered for task {task_name!r}")
        tmpl = task_styles.get(style)
        if tmpl is None:
            raise KeyError(
                f"No prompt style {style!r} for task {task_name!r}. "
                f"Available: {list(task_styles.keys())}"
            )
        return tmpl

    def get_available_styles(self, task_name: str) -> List[str]:
        """Return list of registered styles for *task_name*."""
        return list(self._registry.get(task_name, {}).keys())

    def list_tasks(self) -> List[str]:
        """Return list of all task names that have at least one registered prompt."""
        return list(self._registry.keys())

    def __repr__(self) -> str:
        return f"PromptLibrary(tasks={len(self._registry)}, total_templates={sum(len(v) for v in self._registry.values())})"


def get_prompt_library() -> PromptLibrary:
    """Convenience function returning the shared PromptLibrary singleton."""
    return PromptLibrary()
