"""
Shared helpers for parsing list-valued answers from LLM responses.

Used by: running_average, cumulative_sum, reverse_list, rotate_list,
         interleave_lists, set_intersection, set_difference, moving_average.
"""

import re
import json
from typing import Optional, List, Tuple, Union

from .common import extract_from_boxed_formats, extract_from_code_blocks


def _parse_number_list_from_string(text: str) -> Optional[List[float]]:
    """
    Try to parse a Python-style list literal from text.
    Handles: [1, 2, 3], [1.5, 2.5], {1, 2, 3}, 1 2 3 (space separated).
    """
    if not text:
        return None
    text = text.strip()

    # Try JSON array first
    # Replace curly braces with square brackets for set-like notation
    for attempt in [text, text.replace('{', '[').replace('}', ']')]:
        try:
            result = json.loads(attempt)
            if isinstance(result, list):
                return [float(x) for x in result]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Try bracket extraction with regex
    bracket_patterns = [
        r'\[([^\[\]]*)\]',   # [1, 2, 3]
        r'\{([^{}]*)\}',     # {1, 2, 3}
    ]
    for pattern in bracket_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Take the last match (most likely the final answer)
            inner = matches[-1]
            # Split by comma or whitespace
            parts = re.split(r'[,\s]+', inner.strip())
            try:
                nums = [float(p) for p in parts if p]
                if nums:
                    return nums
            except ValueError:
                pass

    # Try comma-separated numbers (no brackets): "4, 3, 2, 1"
    comma_parts = re.split(r',\s*', text.strip())
    if len(comma_parts) >= 2:
        try:
            nums = [float(p.strip()) for p in comma_parts if p.strip() and re.match(r'^-?\d+(\.\d+)?$', p.strip())]
            if len(nums) == len(comma_parts):
                return nums
        except ValueError:
            pass

    # Try space-separated numbers (no brackets)
    parts = re.split(r'\s+', text.strip())
    if len(parts) >= 2:
        try:
            nums = [float(p) for p in parts if p and re.match(r'^-?\d+(\.\d+)?$', p)]
            if len(nums) == len(parts) and nums:
                return nums
        except ValueError:
            pass

    return None


def parse_list_answer(response: str) -> Tuple[Optional[List[float]], bool]:
    """
    Generic list parser: tries boxed → explicit → code block → last bracket.

    Returns:
        (parsed_list, instruction_followed)
    """
    if not response:
        return None, False

    # 1. Try boxed format
    boxed_raw, instruction_followed = extract_from_boxed_formats(response)
    if boxed_raw is not None:
        result = _parse_number_list_from_string(boxed_raw)
        if result is not None:
            return result, True

    # 2. Try explicit statements like "the result is [1, 2, 3]"
    explicit_patterns = [
        r'(?:result|answer|output|list|sequence|averages?|sums?)\s+(?:is|are|=|:)\s*(\[.*?\])',
        r'(?:result|answer|output|list|sequence|averages?|sums?)\s+(?:is|are|=|:)\s*(\{.*?\})',
        r'=\s*(\[.*?\])\s*$',
        r':\s*(\[.*?\])\s*$',
        # "the answer is 4, 3, 2, 1" (numbers after colon/is with no brackets)
        r'(?:result|answer|output)\s+(?:is|are|=|:)\s*((?:-?\d+(?:\.\d+)?\s*,\s*){2,}-?\d+(?:\.\d+)?)',
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            result = _parse_number_list_from_string(matches[-1])
            if result is not None:
                return result, False

    # 3. Try code blocks
    code = extract_from_code_blocks(response)
    if code:
        result = _parse_number_list_from_string(code)
        if result is not None:
            return result, False

    # 4. Try any bracket in the response (last occurrence)
    bracket_matches = re.findall(r'\[([^\[\]]+)\]', response)
    if bracket_matches:
        result = _parse_number_list_from_string('[' + bracket_matches[-1] + ']')
        if result is not None:
            return result, False

    # 5. Try comma-separated numbers following a keyword
    comma_patterns = [
        # "is/are: 4, 3, 2, 1" or "= 4, 3, 2, 1" at end of line
        r'(?:is|are|=|:)\s*((?:-?\d+(?:\.\d+)?\s*,\s*){1,}-?\d+(?:\.\d+)?)\s*(?:[.\n]|$)',
        # Last line that looks like a comma-separated list of numbers
        r'^((?:-?\d+(?:\.\d+)?\s*,\s*){1,}-?\d+(?:\.\d+)?)\s*$',
    ]
    for pattern in comma_patterns:
        matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
        if matches:
            result = _parse_number_list_from_string(matches[-1])
            if result is not None and len(result) >= 2:
                return result, False

    # 6. Last line: if it contains only numbers and commas, parse it
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    for line in reversed(lines[-3:]):
        clean_line = re.sub(r'[.\s]+$', '', line)
        result = _parse_number_list_from_string(clean_line)
        if result is not None and len(result) >= 2:
            return result, False

    return None, False
