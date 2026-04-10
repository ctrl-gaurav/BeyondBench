"""
Parser for range task answers.

The task asks for max - min (the statistical range).
Models may say: "range is 15", "the range = 15", "\boxed{15}",
"difference between max and min is 15", "15 is the range", etc.
"""

import re
from typing import Optional, Tuple, Union

from .common import (
    extract_from_boxed_formats,
    extract_from_explicit_statements,
    extract_from_latex_math,
    extract_from_code_blocks,
    extract_from_last_line,
    clean_and_convert_to_number,
)

_TASK_PATTERNS = [
    # "the range is 15" / "range = 15"
    r'(?:the\s+)?(?:statistical\s+)?range\s+(?:of (?:the )?(?:list|sequence|numbers?|data)\s+)?(?:is|=|equals|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "range: 15"
    r'\brange\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
    # "15 is the range"
    r'([+-]?\d+(?:\.\d+)?)\s+is\s+(?:the\s+)?(?:statistical\s+)?range',
    # "max - min = 15" or "maximum minus minimum = 15"
    r'(?:max(?:imum)?\s*[-–]\s*min(?:imum)?|maximum\s+minus\s+minimum)\s*=\s*([+-]?\d+(?:\.\d+)?)',
    # "difference between (the) max(imum) and (the) min(imum) is 15"
    r'difference\s+between\s+(?:the\s+)?max(?:imum)?\s+and\s+(?:the\s+)?min(?:imum)?\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?)',
    # "spread is 15"
    r'(?:the\s+)?spread\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_range_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the range (max - min) from an LLM response.

    Args:
        clean_response: The (possibly pre-cleaned) response from the LLM.

    Returns:
        (instruction_followed, parsed_value)
    """
    if not clean_response:
        return False, None

    task_match = _try_task_patterns(clean_response)
    if task_match is not None:
        return True, clean_and_convert_to_number(task_match)

    boxed, followed = extract_from_boxed_formats(clean_response)
    if boxed is not None:
        return followed, clean_and_convert_to_number(boxed)

    explicit, _ = extract_from_explicit_statements(clean_response)
    if explicit is not None:
        return False, clean_and_convert_to_number(explicit)

    latex = extract_from_latex_math(clean_response)
    if latex is not None:
        return False, clean_and_convert_to_number(latex)

    code = extract_from_code_blocks(clean_response)
    if code is not None:
        return False, clean_and_convert_to_number(code)

    last = extract_from_last_line(clean_response)
    if last is not None:
        return False, clean_and_convert_to_number(last)

    return False, None
