"""
Parser for max_adjacent_difference task answers.

The task asks for the maximum absolute difference between any two consecutive elements.
Models may say: "maximum adjacent difference is 8", "largest gap is 8",
"max diff = 8", "\boxed{8}", "8 is the maximum difference between adjacent elements", etc.
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
    # "maximum adjacent difference is 8"
    r'(?:maximum|max|largest|greatest|biggest)\s+(?:adjacent|consecutive|neighboring)\s+(?:difference|gap|distance)\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "max difference between adjacent/consecutive elements is 8"
    r'(?:maximum|max|largest)\s+difference\s+between\s+(?:adjacent|consecutive)\s+(?:element|number|value)s?\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "adjacent difference: 8" / "max adjacent diff: 8"
    r'(?:max(?:imum)?\s+)?adjacent\s+(?:difference|diff|gap)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
    # "8 is the maximum adjacent difference"
    r'([+-]?\d+(?:\.\d+)?)\s+is\s+(?:the\s+)?(?:maximum|max|largest)\s+(?:adjacent|consecutive)\s+(?:difference|gap)',
    # "largest gap between consecutive elements is 8"
    r'(?:largest|greatest|maximum|max)\s+gap\s+(?:between\s+consecutive\s+(?:element|number|value)s?\s+)?(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "max diff = 8"
    r'max(?:imum)?\s+diff(?:erence)?\s*[=:]\s*([+-]?\d+(?:\.\d+)?)',
    # "the biggest jump/step is 8"
    r'(?:biggest|largest|maximum|greatest)\s+(?:jump|step|change|variation)\s+(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)',
    # "max |a[i] - a[i+1]| = 8"
    r'max(?:imum)?\s+\|?a\[i\+?1?\]\s*-\s*a\[i\]\|?\s*=\s*([+-]?\d+(?:\.\d+)?)',
    # "8 is the maximum absolute difference"
    r'([+-]?\d+(?:\.\d+)?)\s+is\s+(?:the\s+)?(?:maximum|max|largest)\s+(?:absolute\s+)?difference',
]


def _try_task_patterns(text: str) -> Optional[str]:
    for pattern in _TASK_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_max_adjacent_difference_answer(clean_response: str) -> Tuple[bool, Optional[Union[int, float]]]:
    """
    Extract the maximum adjacent difference from an LLM response.

    Args:
        clean_response: The (possibly pre-cleaned) response from the LLM.

    Returns:
        (instruction_followed, parsed_value)
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
