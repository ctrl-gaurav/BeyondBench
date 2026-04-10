"""
Parser for second_maximum task answers.

The task asks for the second largest (unique) element in a list.
Models may say: "second maximum", "second largest", "second biggest",
"second highest", "\boxed{7}", "7 is the second largest", etc.
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

# Task-specific keyword patterns (highest confidence, checked first)
_TASK_PATTERNS = [
    # "The second maximum is 7" / "second largest number is 7"
    r'(?:the\s+)?second\s+(?:maximum|largest|biggest|greatest|highest|max)\s+(?:number\s+)?(?:is|=|:|equals)\s*([+-]?\d+(?:\.\d+)?)',
    # "second maximum: 7"
    r'second\s+(?:maximum|largest|biggest|greatest|highest|max)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
    # "7 is the second (maximum|largest|...)"
    r'([+-]?\d+(?:\.\d+)?)\s+is\s+(?:the\s+)?second\s+(?:maximum|largest|biggest|greatest|highest|max)',
    # "2nd maximum is 7" / "2nd largest is 7"
    r'2nd\s+(?:maximum|largest|biggest|greatest|highest|max)\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "second element from the top is 7"
    r'second\s+(?:element|value|number)\s+from\s+(?:the\s+)?top\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?)',
    # "penultimate max is 7"
    r'penultimate\s+(?:max|maximum|largest)\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?)',
    # After sorting descending: "second element is 7"
    r'second\s+element\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_second_maximum_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the second maximum value from an LLM response.

    Tries (in order):
    1. Task-specific keyword patterns ("second largest is …")
    2. Boxed formats (\\boxed{…})
    3. Explicit statements ("the answer is …")
    4. LaTeX math
    5. Code blocks
    6. Last line / last number in text

    Args:
        clean_response: The (possibly pre-cleaned) response from the LLM.

    Returns:
        (instruction_followed, parsed_value)
        instruction_followed is True when a boxed or explicit format was found.
    """
    if not clean_response:
        return False, None

    # 1. Task-specific keyword patterns
    task_match = _try_task_patterns(clean_response)
    if task_match is not None:
        return True, clean_and_convert_to_number(task_match)

    # 2. Boxed formats
    boxed, followed = extract_from_boxed_formats(clean_response)
    if boxed is not None:
        return followed, clean_and_convert_to_number(boxed)

    # 3. Explicit statements
    explicit, _ = extract_from_explicit_statements(clean_response)
    if explicit is not None:
        return False, clean_and_convert_to_number(explicit)

    # 4. LaTeX math
    latex = extract_from_latex_math(clean_response)
    if latex is not None:
        return False, clean_and_convert_to_number(latex)

    # 5. Code blocks
    code = extract_from_code_blocks(clean_response)
    if code is not None:
        return False, clean_and_convert_to_number(code)

    # 6. Last line / last number
    last = extract_from_last_line(clean_response)
    if last is not None:
        return False, clean_and_convert_to_number(last)

    return False, None
