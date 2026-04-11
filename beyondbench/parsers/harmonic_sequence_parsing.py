"""
Parser for harmonic_sequence task.

Handles both fraction (1/7) and decimal (0.142857) answers.
"""

import re
from typing import Optional

from .common import extract_from_boxed_formats


def parse_harmonic_answer(response: str) -> Optional[float]:
    """
    Extract harmonic sequence answer from response.

    Accepts:
    - \\boxed{1/7}
    - \\boxed{0.142857}
    - "The next term is 1/7"
    - "The next term is 0.142857"
    - fraction or decimal in explicit statement
    """
    if not response:
        return None

    # 1. Try boxed format
    raw, found = extract_from_boxed_formats(response)
    if found and raw:
        result = _try_parse_fraction_or_decimal(raw.strip())
        if result is not None:
            return result

    # 2. Scan all boxed occurrences (last one wins for multi-step responses)
    for pattern in [
        r'\\boxed\{([^}]+)\}',
        r'\\fbox\{([^}]+)\}',
        r'\$\\boxed\{([^}]+)\}\$',
    ]:
        matches = re.findall(pattern, response)
        if matches:
            for raw in reversed(matches):
                result = _try_parse_fraction_or_decimal(raw.strip())
                if result is not None:
                    return result

    # 3. Explicit fraction patterns in text
    fraction_patterns = [
        r'(?:next term|answer)[^\d/]*(\d+/\d+)',
        r'(?:is|=)\s*(\d+/\d+)',
        r'(\d+/\d+)\s*(?:$|\.)',
    ]
    for pattern in fraction_patterns:
        m = re.search(pattern, response, re.I)
        if m:
            result = _try_parse_fraction_or_decimal(m.group(1))
            if result is not None:
                return result

    # 4. Decimal patterns
    decimal_patterns = [
        r'(?:next term|answer)[^\d-]*(-?(?:\d+\.\d+|\d+))',
        r'(?:is|=)\s*(-?(?:\d+\.\d+))',
    ]
    for pattern in decimal_patterns:
        m = re.search(pattern, response, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # 5. Last number in response
    nums = re.findall(r'-?(?:\d+/\d+|\d+\.\d+)', response)
    if nums:
        return _try_parse_fraction_or_decimal(nums[-1])

    return None


def _try_parse_fraction_or_decimal(text: str) -> Optional[float]:
    """Try to parse text as a fraction or decimal float."""
    text = text.strip()
    # fraction
    m = re.match(r'^(-?\d+)\s*/\s*(\d+)$', text)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0:
            return num / den
    # decimal or int
    try:
        return float(text)
    except ValueError:
        return None
