"""
SequenceStrategy: Extract next-term answers for sequence completion tasks.

Handles "The next term is X", continuation patterns, negative terms, float
answers, and plain numeric extraction as a fallback.
"""

import re
from typing import Optional
from .boxed_strategy import ParseResult

# Ordered from most specific/confident to least specific.
# Each entry: (regex_pattern, confidence_score)
_SEQUENCE_PATTERNS = [
    # "The next term is X" / "The next number is X"
    (r'(?:the )?next\s+(?:term|number|element|value|item)\s+(?:in the sequence\s+)?(?:is|would be|=)\s*([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)', 0.92),
    # "The following term is X"
    (r'(?:the )?following\s+(?:term|number|element)\s+(?:is|would be)\s*([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)', 0.90),
    # "Therefore/Thus/Hence, the next number is X"
    (r'(?:therefore|thus|hence|so)[,\s]+(?:the )?next\s+(?:term|number|element)?\s*(?:is|=)\s*([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)', 0.88),
    # "answer is X" / "result is X"
    (r'(?:answer|result)\s+(?:is|=)\s*([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)', 0.87),
    # "Continuing the pattern, the next term is X"
    (r'continuing\s+(?:the\s+)?(?:pattern|sequence)[,\s]+(?:the\s+)?(?:next\s+)?(?:term\s+)?(?:is\s+)?([+-]?\d+(?:[.,]\d+)?)', 0.85),
    # "= X" at end of expression (e.g. "F(9) = 34")
    (r'=\s*([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)\s*$', 0.75),
    # "nth term: X" / "term n: X"
    (r'(?:nth|n(?:th|st|rd|nd))\s+term[:\s]+([+-]?\d+(?:[.,]\d+)?)', 0.82),
    # "the missing term is X" / "the missing number is X"
    (r'(?:the\s+)?missing\s+(?:term|number|value)\s+(?:is|=)\s*([+-]?\d+(?:[.,]\d+)?)', 0.88),
    # "predict X" / "predicted value: X"
    (r'predict(?:ed)?\s*(?:value|term|answer)?\s*(?::|is|=)\s*([+-]?\d+(?:[.,]\d+)?)', 0.80),
    # "X is the next term"
    (r'([+-]?\d+(?:[.,]\d+)?)\s+is\s+the\s+next\s+(?:term|number|element)', 0.86),
    # "sequence: ..., X" — last number after comma in a sequence-like list
    (r'(?:sequence|pattern)\s*:?\s*(?:[+-]?\d+[,\s]+){2,}([+-]?\d+(?:[.,]\d+)?)\s*$', 0.70),
    # "? = X" where ? marks the unknown term
    (r'\?\s*=\s*([+-]?\d+(?:[.,]\d+)?)', 0.88),
    # "the answer is X" (generic, lower confidence than sequence-specific)
    (r'(?:the )?answer\s+(?:is|=)\s*([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)', 0.75),
    # "F(n) = X" or "a_n = X" or "T(n) = X"
    (r'[a-zA-Z][\(_]\d+[\)_]?\s*=\s*([+-]?\d+(?:[.,]\d+)?)', 0.80),
    # A lone number on its own line (very last resort, low confidence)
    (r'^\s*([+-]?\d+(?:[.,]\d+)?)\s*$', 0.50),
]

# Pattern for "The next term is approximately X" — strip "approximately"
_APPROX_RE = re.compile(r'\bapproximately\b\s*', re.IGNORECASE)


def _normalize_value(raw: str) -> str:
    """Normalize raw match: strip whitespace, replace comma-decimal with dot."""
    raw = raw.strip()
    # European-style decimal comma: "3,14" → "3.14" (only when single comma present)
    if raw.count(',') == 1 and '.' not in raw:
        parts = raw.split(',')
        if len(parts[1]) <= 4:  # looks like decimal part
            raw = raw.replace(',', '.')
    return raw


class SequenceStrategy:
    """Extract next-term answers for sequence completion tasks."""

    NAME = "sequence"

    def extract(self, text: str, expected_type: str = "number") -> ParseResult:
        # Strip "approximately" before matching
        text_clean = _APPROX_RE.sub('', text)

        for pattern, conf in _SEQUENCE_PATTERNS:
            matches = re.findall(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
            if matches:
                raw = matches[-1]
                if isinstance(raw, tuple):
                    raw = raw[0]
                raw = _normalize_value(raw)
                if raw:
                    return ParseResult(
                        value=raw,
                        confidence=conf,
                        strategy_used=self.NAME,
                        raw_match=raw,
                    )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)
