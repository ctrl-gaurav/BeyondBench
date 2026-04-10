"""
Parser for count_greater_than_previous task answers.

The task asks how many elements (after the first) are strictly greater than
the element immediately before them.
Models may say: "3 elements are greater than the previous",
"count is 3", "there are 3 ascending pairs", "\boxed{3}", etc.
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
    # "3 elements are greater than the previous (element)"
    r'(\d+)\s+(?:element|value|number|integer)s?\s+(?:are|is)\s+greater\s+than\s+(?:the\s+)?previous',
    # "there are 3 elements greater than the previous"
    r'there\s+(?:are|is)\s+(\d+)\s+(?:element|value|number|integer)s?\s+(?:that\s+)?(?:are\s+)?greater\s+than\s+(?:the\s+)?previous',
    # "count of elements greater than previous is 3"
    r'count\s+of\s+(?:element|value|number)s?\s+greater\s+than\s+(?:the\s+)?previous\s+(?:is|=|:)\s*(\d+)',
    # "3 numbers exceed their predecessor"
    r'(\d+)\s+(?:number|element|value)s?\s+(?:exceed|surpass|are\s+larger\s+than|are\s+bigger\s+than)\s+(?:their|the)\s+(?:predecessor|previous)',
    # "number of increasing steps/elements is 3"
    r'(?:number|count|total)\s+of\s+(?:increasing|ascending)\s+(?:step|element|pair|transition)s?\s+(?:is|=|:)\s*(\d+)',
    # "3 ascending/increasing transitions"
    r'(\d+)\s+(?:ascending|increasing)\s+(?:step|element|pair|transition)',
    # "greater_than_previous count: 3"
    r'(?:greater.than.previous|ascending|increasing)\s+count\s*[:=]\s*(\d+)',
    # "3 times an element is greater than its neighbor"
    r'(\d+)\s+time[s]?\s+(?:an?\s+)?element\s+is\s+greater',
    # "3 pairs where next > previous"
    r'(\d+)\s+pair[s]?\s+where\s+(?:next|current)\s+>\s+(?:previous|prev)',
    # "number of elements that increased: 3"
    r'(?:number|count)\s+of\s+elements?\s+that\s+increased\s*[:=]\s*(\d+)',
    # "3 increases" / "3 increments"
    r'(\d+)\s+(?:increase|increment|step\s+up)[s]?',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_count_greater_than_previous_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the count of elements greater than the previous element.

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
