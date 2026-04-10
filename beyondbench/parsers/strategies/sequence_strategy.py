"""
SequenceStrategy: Extract next-term answers for sequence completion tasks.

Handles "The next term is X", continuation patterns, and numeric extraction.
"""

import re
from typing import Optional
from .boxed_strategy import ParseResult

_SEQUENCE_PATTERNS = [
    # "The next term is X"
    (r'(?:the )?next\s+(?:term|number|element|value|item)\s+(?:in the sequence\s+)?(?:is|would be|=)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', 0.90),
    # "The following term is X"
    (r'(?:the )?following\s+(?:term|number|element)\s+(?:is|would be)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', 0.88),
    # "Therefore, the next number is X"
    (r'(?:therefore|thus|hence|so)[,\s]+(?:the )?next\s+(?:term|number|element)?\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', 0.87),
    # "Continuing the pattern, X"
    (r'continuing\s+(?:the\s+)?(?:pattern|sequence)[,\s]+(?:the\s+)?(?:next\s+)?(?:term\s+)?(?:is\s+)?([+-]?\d+(?:\.\d+)?)', 0.83),
    # "= X" at end of sequence expression
    (r'=\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$', 0.72),
    # "nth term: X" or "term n: X"
    (r'(?:nth|n(?:th|st|rd|nd))\s+term[:\s]+([+-]?\d+(?:\.\d+)?)', 0.80),
]


class SequenceStrategy:
    """Extract next-term answers for sequence completion tasks."""

    NAME = "sequence"

    def extract(self, text: str, expected_type: str = "number") -> ParseResult:
        for pattern, conf in _SEQUENCE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                raw = matches[-1]
                if isinstance(raw, tuple):
                    raw = raw[0]
                raw = raw.strip()
                if raw:
                    return ParseResult(
                        value=raw,
                        confidence=conf,
                        strategy_used=self.NAME,
                        raw_match=raw,
                    )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)
