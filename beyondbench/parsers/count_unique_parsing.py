"""
Parser for count_unique task answers.

The task asks how many unique/distinct elements are in the list.
Models may say: "4 unique elements", "there are 4 distinct values",
"count of unique elements is 4", "\boxed{4}", etc.
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
    # "there are 4 unique/distinct elements/values"
    r'there\s+(?:are|is)\s+(\d+)\s+(?:unique|distinct|different)',
    # "4 unique/distinct elements/values in the list"
    r'(\d+)\s+(?:unique|distinct|different)\s+(?:element|value|number|integer)',
    # "count of unique/distinct elements is 4"
    r'count\s+of\s+(?:unique|distinct|different)\s+(?:element|value|number|integer)?s?\s+(?:is|=|:)\s*(\d+)',
    # "number of unique/distinct elements is 4"
    r'(?:number|total|quantity|amount)\s+of\s+(?:unique|distinct|different)\s+(?:element|value|number|integer)?s?\s+(?:is|=|:)\s*(\d+)',
    # "unique count: 4" / "distinct count: 4"
    r'(?:unique|distinct|different)\s+(?:count|number|total)\s*[:=]\s*(\d+)',
    # "4 elements are unique/distinct"
    r'(\d+)\s+(?:element|value|number|integer)s?\s+(?:are|is)\s+(?:unique|distinct)',
    # "uniqueness: 4" (unusual but handled)
    r'uniqueness\s*[:=]\s*(\d+)',
    # "4 of them are unique"
    r'(\d+)\s+of\s+them\s+(?:are|is)\s+(?:unique|distinct)',
    # "set has 4 elements" (set = unique elements)
    r'set\s+has\s+(\d+)\s+(?:element|value|member)',
    # "cardinality is 4"
    r'cardinality\s+(?:is|=|:)\s*(\d+)',
    # "found 4 unique"
    r'found\s+(\d+)\s+(?:unique|distinct)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_count_unique_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the count of unique elements from an LLM response.

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
