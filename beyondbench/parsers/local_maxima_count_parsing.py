"""
Parser for local_maxima_count task answers.

A local maximum is an element strictly greater than both its neighbors.
The task asks how many local maxima the list has.
Models may say: "2 local maxima", "there are 2 local maxima",
"count is 2", "\boxed{2}", "2 peaks", etc.
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
    # "there are 2 local maxima"
    r'there\s+(?:are|is)\s+(\d+)\s+local\s+(?:maxima|maximum|max)',
    # "2 local maxima/maximums in the list"
    r'(\d+)\s+local\s+(?:maxima|maximum|max)',
    # "count of local maxima is 2"
    r'count\s+of\s+local\s+(?:maxima|maximum|max)\s+(?:is|=|:)\s*(\d+)',
    # "number of local maxima is 2"
    r'(?:number|total|quantity|amount)\s+of\s+local\s+(?:maxima|maximum|max)\s+(?:is|=|:)\s*(\d+)',
    # "local maxima count: 2"
    r'local\s+(?:maxima|maximum|max)\s+(?:count|number|total)\s*[:=]\s*(\d+)',
    # "local maxima: 2"
    r'local\s+(?:maxima|maximum|max)\s*[:=]\s*(\d+)',
    # "2 peaks" / "2 peak elements"
    r'(\d+)\s+peak(?:\s+element)?s?',
    # "there are 2 peaks"
    r'there\s+(?:are|is)\s+(\d+)\s+peak',
    # "peak count: 2"
    r'peak\s+(?:count|number|total)\s*[:=]\s*(\d+)',
    # "2 of them are local maxima"
    r'(\d+)\s+of\s+them\s+(?:are|is)\s+(?:a\s+)?local\s+(?:maxima|maximum|max)',
    # "found 2 local maxima"
    r'found\s+(\d+)\s+local\s+(?:maxima|maximum|max)',
    # "2 elements are local maxima"
    r'(\d+)\s+(?:element|value|number)s?\s+(?:are|is)\s+(?:a\s+)?local\s+(?:maxima|maximum|max)',
    # "2 interior peaks"
    r'(\d+)\s+interior\s+(?:peak|maximum|maxima)',
    # "maxima count: 2"
    r'maxima\s+(?:count|number)\s*[:=]\s*(\d+)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_local_maxima_count_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the count of local maxima from an LLM response.

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
