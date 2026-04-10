"""
Parser for sum_of_digits task answers.

The task asks for the sum of the decimal digits of a given integer.
E.g. sum_of_digits(1234) = 1+2+3+4 = 10.
Models may say: "sum of digits is 10", "digit sum = 10",
"1+2+3+4 = 10", "\boxed{10}", "10", etc.
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
    # "sum of (the) digits is 10" / "digit sum is 10"
    r'(?:sum\s+of\s+(?:the\s+)?digits?|digit\s+sum)\s+(?:is|=|:|equals)\s*([+-]?\d+(?:\.\d+)?)',
    # "digits sum to 10" / "digits add up to 10"
    r'digits?\s+(?:sum\s+to|add\s+up\s+to|total\s+to)\s+([+-]?\d+(?:\.\d+)?)',
    # "sum: 10" / "digit sum: 10"
    r'(?:sum|digit\s+sum|total)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
    # "10 is the sum of digits"
    r'([+-]?\d+(?:\.\d+)?)\s+is\s+(?:the\s+)?(?:sum\s+of\s+(?:the\s+)?digits?|digit\s+sum)',
    # "sum of all digits: 10"
    r'sum\s+of\s+all\s+digits?\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
    # "total digit sum: 10"
    r'total\s+digit\s+sum\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
    # "digital root is 10" (not quite the same, but sometimes confused)
    r'digital\s+(?:root|sum)\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # Arithmetic expression result: "1 + 2 + 3 + 4 = 10"
    r'(?:\d+\s*\+\s*)+\d+\s*=\s*([+-]?\d+(?:\.\d+)?)',
    # "answer: 10" (generic but common)
    r'answer\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_sum_of_digits_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the sum of digits from an LLM response.

    Args:
        clean_response: The (possibly pre-cleaned) response from the LLM.

    Returns:
        (instruction_followed, digit_sum)
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
