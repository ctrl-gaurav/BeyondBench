"""
Parser for count_perfect_squares task answers.

The task asks how many numbers in a list are perfect squares (0, 1, 4, 9, 16, …).
Models may say: "3 perfect squares", "there are 3 perfect squares",
"count is 3", "\boxed{3}", etc.
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
    # "there are 3 perfect squares"
    r'there\s+(?:are|is)\s+(\d+)\s+perfect\s+square',
    # "3 perfect squares in the list"
    r'(\d+)\s+perfect\s+square',
    # "count of perfect squares is 3"
    r'count\s+of\s+perfect\s+squares?\s+(?:is|=|:)\s*(\d+)',
    # "number of perfect squares is 3"
    r'(?:number|total|quantity|amount)\s+of\s+perfect\s+squares?\s+(?:is|=|:)\s*(\d+)',
    # "perfect square count: 3"
    r'perfect\s+square\s+(?:count|number|total)\s*[:=]\s*(\d+)',
    # "3 of them are perfect squares"
    r'(\d+)\s+of\s+them\s+(?:are|is)\s+perfect\s+square',
    # "found 3 perfect squares"
    r'found\s+(\d+)\s+perfect\s+square',
    # "perfect squares: 3"
    r'perfect\s+squares?\s*[:=]\s*(\d+)',
    # "squares: 3"
    r'squares?\s*[:=]\s*(\d+)',
    # "3 numbers are perfect squares"
    r'(\d+)\s+(?:number|element|value|integer)s?\s+(?:are|is)\s+(?:a\s+)?perfect\s+square',
    # "3 square numbers"
    r'(\d+)\s+square\s+(?:number|integer)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_count_perfect_squares_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the count of perfect squares from an LLM response.

    Args:
        clean_response: The (possibly pre-cleaned) response from the LLM.

    Returns:
        (instruction_followed, count)
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
