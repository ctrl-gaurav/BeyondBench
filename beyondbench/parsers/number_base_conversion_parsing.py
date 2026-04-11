"""
Parser for number_base_conversion task.

Handles string answers (binary/hex) or integer answers (decimal).
"""

import re
from typing import Optional

from .common import extract_from_boxed_formats


def parse_base_conversion_answer(response: str, to_base: int) -> Optional[str]:
    """
    Extract base conversion answer from response.

    Returns a string (uppercase for hex, plain digits for binary/decimal).
    """
    if not response:
        return None

    # 1. Try boxed format
    raw, found = extract_from_boxed_formats(response)
    if found and raw:
        cleaned = _clean_base_answer(raw.strip(), to_base)
        if cleaned is not None:
            return cleaned

    # 2. All boxed occurrences (last wins)
    for pattern in [r'\\boxed\{([^}]+)\}', r'\$\\boxed\{([^}]+)\}\$']:
        matches = re.findall(pattern, response)
        for raw in reversed(matches):
            cleaned = _clean_base_answer(raw.strip(), to_base)
            if cleaned is not None:
                return cleaned

    # 3. Explicit statements
    if to_base == 2:
        patterns = [
            r'(?:binary|answer)[^\d]*([01]+)',
            r'(?:is|=)\s*([01]+)',
        ]
    elif to_base == 16:
        patterns = [
            r'(?:hexadecimal|hex|answer)[^\dA-Fa-f]*([0-9A-Fa-f]+)',
            r'(?:is|=)\s*([0-9A-Fa-f]+)',
            r'0x([0-9A-Fa-f]+)',
        ]
    else:
        patterns = [
            r'(?:decimal|answer)[^\d]*(\d+)',
            r'(?:is|=)\s*(\d+)',
        ]

    for pat in patterns:
        m = re.search(pat, response, re.I)
        if m:
            cleaned = _clean_base_answer(m.group(1).strip(), to_base)
            if cleaned is not None:
                return cleaned

    # 4. Last plausible token
    if to_base == 2:
        tokens = re.findall(r'\b[01]{2,16}\b', response)
        if tokens:
            return tokens[-1]
    elif to_base == 16:
        tokens = re.findall(r'\b[0-9A-Fa-f]{1,8}\b', response)
        if tokens:
            return tokens[-1].upper()
    else:
        tokens = re.findall(r'\b\d+\b', response)
        if tokens:
            return tokens[-1]

    return None


def _clean_base_answer(text: str, to_base: int) -> Optional[str]:
    """Validate and normalise a base-specific token."""
    # Strip common prefixes
    text = text.strip()
    if to_base == 16 and text.lower().startswith('0x'):
        text = text[2:]
    if to_base == 2 and text.lower().startswith('0b'):
        text = text[2:]

    if not text:
        return None

    if to_base == 2:
        if re.fullmatch(r'[01]+', text):
            return text
        return None
    elif to_base == 16:
        if re.fullmatch(r'[0-9A-Fa-f]+', text):
            return text.upper()
        return None
    else:  # decimal
        if re.fullmatch(r'\d+', text):
            return text
        return None
