"""
Parser for count_palindromic task answers.

The task asks how many numbers in a list are palindromes (read the same forwards
and backwards: 121, 1, 11, 1331, etc.; single digits are always palindromes).
Models may say: "3 palindromic numbers", "there are 3 palindromes",
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
    # "there are 3 palindromes / palindromic numbers"
    r'there\s+(?:are|is)\s+(\d+)\s+palindrom(?:e|ic|s)',
    # "3 palindromes / palindromic numbers in the list"
    r'(\d+)\s+palindrom(?:e|ic|s)',
    # "count of palindromes is 3"
    r'count\s+of\s+palindrom(?:e|ic|s)\s+(?:number|value|integer)?s?\s+(?:is|=|:)\s*(\d+)',
    # "number/total of palindromes is 3"
    r'(?:number|total|quantity|amount)\s+of\s+palindrom(?:e|ic|s)(?:\s+(?:number|value|integer))?s?\s+(?:is|=|:)\s*(\d+)',
    # "palindrome count: 3"
    r'palindrom(?:e|ic)?\s+count\s*[:=]\s*(\d+)',
    # "palindromic count: 3"
    r'palindromic\s+(?:count|number|total)\s*[:=]\s*(\d+)',
    # "3 of them are palindromes"
    r'(\d+)\s+of\s+them\s+(?:are|is)\s+palindrom(?:e|ic)',
    # "found 3 palindromes"
    r'found\s+(\d+)\s+palindrom(?:e|ic|s)',
    # "palindromes: 3"
    r'palindromes?\s*[:=]\s*(\d+)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_count_palindromic_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the count of palindromic numbers from an LLM response.

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
