"""
Parser for element_frequency task answers.
Expected output: [[val, count], [val, count], ...] sorted by val.
"""

import re
import json
from typing import Optional, List, Tuple

from .common import extract_from_boxed_formats, extract_from_code_blocks


def _parse_nested_list(text: str) -> Optional[List[List[float]]]:
    """Try to parse [[val, count], ...] from text."""
    if not text:
        return None
    text = text.strip()

    # Try JSON first
    try:
        result = json.loads(text)
        if isinstance(result, list) and all(isinstance(r, list) and len(r) == 2 for r in result):
            return [[float(r[0]), float(r[1])] for r in result]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try nested bracket regex: [[a, b], [c, d], ...]
    # Find outer brackets
    outer_match = re.search(r'\[\s*(\[.*\])\s*\]', text, re.DOTALL)
    if outer_match:
        inner = outer_match.group(0)
        try:
            result = json.loads(inner)
            if isinstance(result, list):
                return [[float(r[0]), float(r[1])] for r in result]
        except (json.JSONDecodeError, TypeError, ValueError, IndexError):
            pass
        # Manual extraction of pairs
        pairs = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', inner)
        if pairs:
            return [[float(a), float(b)] for a, b in pairs]

    # Just find all [val, count] pairs anywhere in text
    pairs = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text)
    if pairs:
        return [[float(a), float(b)] for a, b in pairs]

    return None


def parse_element_frequency_answer(response: str) -> Tuple[Optional[List[List[float]]], bool]:
    """
    Extract [[val, count], ...] frequency table from LLM response.

    Returns:
        (parsed_list, instruction_followed)
    """
    if not response:
        return None, False

    # 1. Try boxed format
    boxed, followed = extract_from_boxed_formats(response)
    if boxed is not None:
        result = _parse_nested_list(boxed)
        if result is not None:
            return result, True

    # 2. Try code blocks
    code = extract_from_code_blocks(response)
    if code:
        result = _parse_nested_list(code)
        if result is not None:
            return result, False

    # 3. Try to find [[...]] anywhere in the response
    # Find the outermost nested list
    nested_patterns = [
        r'\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\]',
    ]
    for pattern in nested_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            result = _parse_nested_list(matches[-1])
            if result is not None:
                return result, False

    return None, False
