"""
Parser for parity_check task answers.
Expected output: 'even' or 'odd'.
"""

import re
from typing import Optional, Tuple

from .common import extract_from_boxed_formats, extract_from_explicit_statements


def parse_parity_check_answer(response: str) -> Tuple[Optional[str], bool]:
    """
    Extract 'even' or 'odd' from an LLM response.

    Returns:
        (parsed_str, instruction_followed)
    """
    if not response:
        return None, False

    # 1. Try boxed format first
    boxed, followed = extract_from_boxed_formats(response)
    if boxed is not None:
        lower = boxed.strip().lower()
        if lower in ("even", "odd"):
            return lower, True

    # 2. Task-specific patterns
    task_patterns = [
        r'product\s+is\s+(even|odd)',
        r'result\s+is\s+(even|odd)',
        r'answer\s+is\s+(even|odd)',
        r'(even|odd)\s+number',
        r'(even|odd)\s+product',
        r'\b(even|odd)\b',
    ]
    for pattern in task_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].lower(), False

    return None, False
