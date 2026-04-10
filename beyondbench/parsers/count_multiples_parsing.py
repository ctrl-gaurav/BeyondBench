"""
Parser for count_multiples task answers.

The task asks how many numbers in a list are multiples of a given divisor k
(i.e. n % k == 0).
Models may say: "3 multiples of 4", "there are 3 multiples",
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
    # "there are 3 multiples (of k)"
    r'there\s+(?:are|is)\s+(\d+)\s+multiple',
    # "3 multiples of k in the list"
    r'(\d+)\s+multiple[s]?(?:\s+of\s+\d+)?',
    # "count of multiples (of k) is 3"
    r'count\s+of\s+multiples?\s+(?:of\s+\d+\s+)?(?:is|=|:)\s*(\d+)',
    # "number of multiples is 3"
    r'(?:number|total|quantity|amount)\s+of\s+multiples?\s+(?:of\s+\d+\s+)?(?:is|=|:)\s*(\d+)',
    # "multiple count: 3"
    r'multiple\s+(?:count|number|total)\s*[:=]\s*(\d+)',
    # "multiples: 3"
    r'multiples?\s*[:=]\s*(\d+)',
    # "3 of them are multiples of k"
    r'(\d+)\s+of\s+them\s+(?:are|is)\s+(?:a\s+)?multiple',
    # "found 3 multiples"
    r'found\s+(\d+)\s+multiple',
    # "divisible by k: 3 numbers"
    r'divisible\s+by\s+\d+\s*:\s*(\d+)',
    # "3 numbers are divisible by k"
    r'(\d+)\s+(?:number|element|value|integer)s?\s+(?:are|is)\s+divisible',
    # "count of numbers divisible by k is 3"
    r'count\s+of\s+(?:number|element|value)s?\s+divisible\s+by\s+\d+\s+(?:is|=|:)\s*(\d+)',
    # "3 are divisible"
    r'(\d+)\s+(?:are|is)\s+divisible',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_count_multiples_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the count of multiples from an LLM response.

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
