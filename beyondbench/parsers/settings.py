"""
Global parser settings for BeyondBench.

These settings control which parser mode is used across all tasks.
Set via CLI flag --parser=unified|legacy, or programmatically.
"""

# Current parser mode: "unified" uses UnifiedParser, "legacy" uses original per-task parsers
PARSER_MODE: str = "unified"


def set_parser_mode(mode: str) -> None:
    """
    Set the global parser mode.

    Args:
        mode: "unified" or "legacy"
    """
    global PARSER_MODE
    if mode not in ("unified", "legacy"):
        raise ValueError(f"Invalid parser mode: {mode!r}. Must be 'unified' or 'legacy'.")
    PARSER_MODE = mode


def get_parser_mode() -> str:
    """Return the current global parser mode."""
    return PARSER_MODE
