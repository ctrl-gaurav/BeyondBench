"""
Parser for logical_operations task.

Handles True/False, 1/0, yes/no, true/false in various formats.
"""

import re
from typing import Optional

from .common import extract_from_boxed_formats


def parse_logical_answer(response: str) -> Optional[bool]:
    """
    Extract True/False answer from logical operations response.

    Accepts:
    - \\boxed{True} / \\boxed{False}
    - \\boxed{true} / \\boxed{false}
    - \\boxed{1} / \\boxed{0}
    - \\boxed{yes} / \\boxed{no}
    - "The answer is True/False"
    - "true"/"false" as standalone word
    """
    if not response:
        return None

    # 1. Try boxed format
    raw, found = extract_from_boxed_formats(response)
    if found and raw:
        result = _parse_bool_token(raw.strip())
        if result is not None:
            return result

    # 2. All boxed occurrences (last wins)
    for pattern in [r'\\boxed\{([^}]+)\}', r'\$\\boxed\{([^}]+)\}\$']:
        matches = re.findall(pattern, response, re.I)
        for raw in reversed(matches):
            result = _parse_bool_token(raw.strip())
            if result is not None:
                return result

    # 3. Explicit statements
    stmt_patterns = [
        r'(?:answer|result|value)[^\w]*(true|false|yes|no|1|0)\b',
        r'(?:is|=)\s*(true|false|yes|no|1|0)\b',
        r'^(true|false)\b',
    ]
    for pat in stmt_patterns:
        m = re.search(pat, response, re.I | re.M)
        if m:
            result = _parse_bool_token(m.group(1))
            if result is not None:
                return result

    # 4. Look for True/False anywhere in response (last occurrence)
    tokens = re.findall(r'\b(true|false|yes|no)\b', response, re.I)
    if tokens:
        return _parse_bool_token(tokens[-1])

    return None


def _parse_bool_token(token: str) -> Optional[bool]:
    """Convert a token to bool."""
    t = token.strip().lower()
    if t in ('true', 'yes', '1'):
        return True
    if t in ('false', 'no', '0'):
        return False
    return None
