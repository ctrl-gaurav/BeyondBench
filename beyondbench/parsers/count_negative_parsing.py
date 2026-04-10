"""
Parser for count_negative task answers.

The task asks how many numbers in a list are strictly negative (< 0).
Models may say: "3 negative numbers", "count of negatives is 3",
"there are 3 negative numbers", "\boxed{3}", etc.
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
    # "there are 3 negative numbers"
    r'there\s+(?:are|is)\s+(\d+)\s+negative',
    # "3 negative numbers/values/integers in the list"
    r'(\d+)\s+negative\s+(?:number|value|integer|element)',
    # "count of negative(s) is 3"
    r'count\s+of\s+negative(?:s|s\s+(?:number|value|integer|element))?\s+(?:is|=|:)\s*(\d+)',
    # "number of negative(s) is 3"
    r'(?:number|total|quantity|amount)\s+of\s+negative\s+(?:number|value|integer|element|s)?\s+(?:is|=|:)\s*(\d+)',
    # "negative count: 3" / "negative numbers: 3"
    r'negative\s+(?:count|number|total|numbers?|values?|integers?|elements?)\s*[:=]\s*(\d+)',
    # "3 numbers are negative"
    r'(\d+)\s+(?:number|value|integer|element)s?\s+(?:are|is)\s+negative',
    # "negatives: 3"
    r'negatives?\s*[:=]\s*(\d+)',
    # "count: 3" after context about negatives
    r'(?:negative.*?)?count\s*[:=]\s*(\d+)',
    # "3 of them are negative"
    r'(\d+)\s+of\s+them\s+(?:are|is)\s+negative',
    # "found 3 negative"
    r'found\s+(\d+)\s+negative',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_count_negative_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the count of negative numbers from an LLM response.

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
