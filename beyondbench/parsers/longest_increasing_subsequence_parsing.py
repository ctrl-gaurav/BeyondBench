"""
Parser for longest_increasing_subsequence task answers.

The task asks for the *length* of the longest strictly increasing subsequence (LIS).
Models may say: "length of LIS is 4", "LIS length = 4", "the LIS has 4 elements",
"longest increasing subsequence has length 4", "\boxed{4}", etc.
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
    # "length of (the) LIS is 4" / "LIS length = 4"
    r'(?:length\s+of\s+(?:the\s+)?LIS|LIS\s+length|LIS\s+has\s+length)\s*(?:is|=|:)\s*(\d+)',
    # "length of the longest increasing subsequence is 4"
    r'length\s+of\s+(?:the\s+)?(?:longest\s+)?increasing\s+subsequence\s+(?:is|=|:)\s*(\d+)',
    # "longest increasing subsequence has length 4"
    r'(?:longest\s+)?increasing\s+subsequence\s+has\s+(?:a\s+)?length\s+(?:of\s+)?(\d+)',
    # "the LIS is 4 elements long"
    r'(?:the\s+)?LIS\s+is\s+(\d+)\s+element',
    # "LIS: 4" / "LIS = 4"
    r'\bLIS\s*[:=]\s*(\d+)',
    # "longest increasing subsequence: 4"
    r'longest\s+increasing\s+subsequence\s*[:=]\s*(\d+)',
    # "LIS length: 4"
    r'LIS\s+(?:length|size|count)\s*[:=]\s*(\d+)',
    # "the answer is 4 (LIS)" or "4 (length of LIS)"
    r'(\d+)\s*\(?\s*(?:LIS|longest\s+increasing)',
    # "4 is the length of the LIS"
    r'(\d+)\s+is\s+(?:the\s+)?(?:length\s+of\s+(?:the\s+)?LIS|LIS\s+length)',
    # "4 is the length of the longest increasing subsequence"
    r'(\d+)\s+is\s+(?:the\s+)?length\s+of\s+(?:the\s+)?(?:longest\s+)?increasing\s+subsequence',
    # "size of LIS is 4"
    r'size\s+of\s+(?:the\s+)?LIS\s+(?:is|=|:)\s*(\d+)',
    # "maximum length is 4" (when context is LIS)
    r'(?:maximum|max)\s+length\s+(?:is|=|:)\s*(\d+)',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_longest_increasing_subsequence_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the length of the longest increasing subsequence from an LLM response.

    Args:
        clean_response: The (possibly pre-cleaned) response from the LLM.

    Returns:
        (instruction_followed, length)
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
