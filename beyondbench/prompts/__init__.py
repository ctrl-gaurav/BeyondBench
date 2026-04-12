"""BeyondBench prompt engineering module.

Exports:
    PromptTemplate  — a named, style-tagged prompt with variable substitution.
    PromptLibrary   — singleton registry of prompt templates per task.
    get_prompt_library — convenience accessor for the shared singleton.
"""

from .template import PromptTemplate
from .library import PromptLibrary, get_prompt_library

__all__ = ["PromptTemplate", "PromptLibrary", "get_prompt_library"]
