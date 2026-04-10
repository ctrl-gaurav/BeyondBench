"""
Pre-built model behavior profiles for BeyondBench adaptive parsing.

These profiles encode known output format statistics for popular models,
enabling the UnifiedParser to reorder extraction strategies without needing
to run a live calibration pass.

Usage:
    from beyondbench.utils.model_profiles import load_builtin_profile
    profile = load_builtin_profile("Qwen/Qwen2.5-1.5B-Instruct")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_PROFILES_DIR = Path(__file__).parent

# Mapping of lowercased model_id substrings to profile JSON filenames.
# Checked in order — first match wins.
_BUILTIN_MAP = [
    ("qwen2.5-0.5b",     "qwen2.5-0.5b-instruct.json"),
    ("qwen2.5-1.5b",     "qwen2.5-1.5b-instruct.json"),
    ("qwen2.5-3b",       "qwen2.5-3b-instruct.json"),
    ("qwen2.5-7b",       "qwen2.5-7b-instruct.json"),
    ("llama-3.2-1b",     "llama-3.2-1b-instruct.json"),
    ("llama-3.2-3b",     "llama-3.2-3b-instruct.json"),
    ("phi-3.5-mini",     "phi-3.5-mini-instruct.json"),
    ("phi-3.5",          "phi-3.5-mini-instruct.json"),
    ("mistral-7b",       "mistral-7b-instruct.json"),
    ("gemma-2-2b",       "gemma-2-2b-it.json"),
    ("gemma-2-9b",       "gemma-2-9b-it.json"),
    ("gpt-4o-mini",      "gpt-4o-mini.json"),
    ("gpt-4o",           "gpt-4o.json"),
    ("gemini-2.5-flash", "gemini-2.5-flash.json"),
    ("claude-sonnet-4",  "claude-sonnet-4.json"),
]


def load_builtin_profile(model_id: str) -> Optional[dict]:
    """
    Load a pre-built profile dict for *model_id*, or None if not found.

    Args:
        model_id: HuggingFace path or API model name.

    Returns:
        Raw profile dict, or None.
    """
    lowered = model_id.lower()
    for key, filename in _BUILTIN_MAP:
        if key in lowered:
            path = _PROFILES_DIR / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
    return None


def list_builtin_models() -> list:
    """Return list of model_id strings with pre-built profiles."""
    return [key for key, _ in _BUILTIN_MAP]
