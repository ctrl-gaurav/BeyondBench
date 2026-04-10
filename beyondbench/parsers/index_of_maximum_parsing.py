"""
Parser for index_of_maximum task answers.

The task asks for the 0-based index of the maximum element.
Models may say: "index 3", "index of maximum is 3", "at position 3",
"at index 3 (0-based)", "\boxed{3}", etc.
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
    # "the index of the maximum is 3" / "index of max is 3"
    r'(?:the\s+)?index\s+of\s+(?:the\s+)?(?:maximum|max|largest|biggest|highest)\s+(?:element\s+|value\s+|number\s+)?(?:is|=|:)\s*(\d+)',
    # "maximum is at index 3" / "maximum occurs at index 3"
    r'(?:maximum|max|largest|biggest|highest)\s+(?:element\s+|value\s+|number\s+)?(?:is\s+)?(?:at|located\s+at|found\s+at|occurs?\s+at)\s+(?:index\s+|position\s+)?(\d+)',
    # "index: 3" / "position: 3"
    r'(?:index|position|location|idx)\s*[:=]\s*(\d+)',
    # "at position 3 (0-indexed)" / "position 3"
    r'(?:at\s+)?position\s+(\d+)',
    # "index 3 (0-based)"
    r'\bindex\s+(\d+)',
    # "3rd element" → 2, "4th element" → 3 (ordinal position, 1-based to 0-based)
    # Not converted here; left for the caller
    # "the 4th element" / "element at index 3"
    r'element\s+at\s+(?:index\s+)?(\d+)',
    # "located at position 3"
    r'located\s+at\s+(?:index\s+|position\s+)?(\d+)',
    # "3 is the index"
    r'(\d+)\s+is\s+(?:the\s+)?(?:index|position)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_index_of_maximum_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the 0-based index of the maximum element from an LLM response.

    Args:
        clean_response: The (possibly pre-cleaned) response from the LLM.

    Returns:
        (instruction_followed, parsed_index)
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
