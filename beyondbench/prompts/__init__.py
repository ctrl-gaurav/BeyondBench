"""BeyondBench prompt engineering module.

Exports:
    PromptTemplate  — a named, style-tagged prompt with variable substitution.
    PromptLibrary   — singleton registry of prompt templates per task.
    get_prompt_library — convenience accessor for the shared singleton.
    FewShotGenerator — dynamic few-shot example generation (contamination-safe).
"""

from .template import PromptTemplate
from .library import PromptLibrary, get_prompt_library
from .few_shot_generator import FewShotGenerator

__all__ = ["PromptTemplate", "PromptLibrary", "get_prompt_library", "FewShotGenerator"]
