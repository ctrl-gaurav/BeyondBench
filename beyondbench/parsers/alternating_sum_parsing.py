"""
Parser for alternating_sum task answers.

The task computes the alternating sum: a[0] - a[1] + a[2] - a[3] + ...
The result can be negative, zero, or positive.
Models may say: "alternating sum is -3", "the result is -3",
"-3", "\boxed{-3}", "result: -3", etc.

Key challenge: correctly extracting negative values.
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
    # "alternating sum is -3" / "alternating sum = -3"
    r'alternating\s+sum\s+(?:is|=|:|equals)\s*([+-]?\d+(?:\.\d+)?)',
    # "alternating sum: -3"
    r'alternating\s+sum\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
    # "-3 is the alternating sum"
    r'([+-]?\d+(?:\.\d+)?)\s+is\s+(?:the\s+)?alternating\s+sum',
    # "result of alternating sum is -3"
    r'result\s+of\s+(?:the\s+)?alternating\s+sum\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "alternating sum (of the list) = -3"
    r'alternating\s+sum\s+(?:of\s+(?:the\s+)?(?:list|sequence|array)\s+)?(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "(+/-) sum is -3"
    r'(?:\+/-|plus(?:\s+minus)?|minus(?:\s+plus)?)\s+sum\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "sum with alternating signs is -3"
    r'sum\s+with\s+alternating\s+signs?\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # Arithmetic result: "a - b + c - d = -3"
    r'[+-]?\d+(?:\s*[-+]\s*\d+)+\s*=\s*([+-]?\d+(?:\.\d+)?)',
    # Generic "result/answer is -3" (allow negative)
    r'(?:result|answer|value|output)\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # Standalone negative number on its own line (crucial for alternating sum)
    r'^\s*([+-]\d+(?:\.\d+)?)\s*$',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        flags = re.IGNORECASE | re.MULTILINE
        matches = re.findall(pattern, text, flags)
        if matches:
            return matches[-1].strip()
    return None


def parse_alternating_sum_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the alternating sum from an LLM response.

    Special care is taken to handle negative results correctly.

    Args:
        clean_response: The (possibly pre-cleaned) response from the LLM.

    Returns:
        (instruction_followed, alternating_sum)
    """
    if not clean_response:
        return False, None

    task_match = _try_task_patterns(clean_response)
    if task_match is not None:
        return True, clean_and_convert_to_number(task_match)

    # For alternating sums, boxed negative numbers are common: \boxed{-3}
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
