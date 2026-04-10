"""
Parser for sum_of_max_indices task answers.

This task asks for the sum of values at positions where the maximum value appears
(or sum over indices where a certain condition holds).
Models may say: "sum at max indices is 12", "\boxed{12}", "12", etc.
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
    # "sum at max indices is 12"
    r'sum\s+(?:at|of|from)\s+(?:the\s+)?max(?:imum)?\s+(?:index|indices|positions?)\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "sum of values at maximum positions is 12"
    r'sum\s+of\s+(?:the\s+)?values?\s+at\s+(?:the\s+)?(?:maximum|max|highest)\s+(?:position|index)\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "sum is 12" (when it's the primary topic)
    r'(?:the\s+)?sum\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "total is 12"
    r'(?:the\s+)?total\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "12 is the sum"
    r'([+-]?\d+(?:\.\d+)?)\s+is\s+(?:the\s+)?sum',
    # "sum of max indices: 12"
    r'sum\s+of\s+max\s+(?:index|indices)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
    # "result: 12"
    r'(?:result|answer|value)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_sum_of_max_indices_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the sum at max indices from an LLM response.

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
