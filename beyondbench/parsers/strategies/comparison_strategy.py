"""
ComparisonStrategy: Extract comparison results (greater than, less than, equal to).

Handles symbols >, <, =  and natural language variants.
"""

import re
from typing import Optional
from .boxed_strategy import ParseResult

_NORMALIZE_MAP = {
    "greater": "greater than",
    "more": "greater than",
    "larger": "greater than",
    "bigger": "greater than",
    "exceeds": "greater than",
    "above": "greater than",
    ">": "greater than",
    "less": "less than",
    "smaller": "less than",
    "lower": "less than",
    "below": "less than",
    "fewer": "less than",
    "<": "less than",
    "equal": "equal to",
    "same": "equal to",
    "identical": "equal to",
    "=": "equal to",
}

# (pattern, confidence)
_COMPARISON_PATTERNS = [
    # Direct comparison terms
    (r'\b(greater than|less than|equal to)\b', 0.90),
    # Symbol patterns: "42 > 37" or "X > Y"
    (r'\b\d+\s*([><])\s*\d+', 0.85),
    # "Number 1 is X than Number 2"
    (r'(?:number\s*1|first number|num\s*1)\s+is\s+(greater|less|equal|larger|smaller|bigger|lower|more|fewer|same)', 0.88),
    # "A is greater/less/equal to B"
    (r'\bis\s+(greater|less|larger|smaller|bigger|lower|more|fewer|equal|same)\b', 0.82),
    # "X exceeds Y"
    (r'\b(exceeds|above|below)\b', 0.78),
    # Reversed: "Number 2 is greater than Number 1" — needs inversion
    (r'(?:number\s*2|second number|num\s*2)\s+is\s+(greater|less|equal|larger|smaller|bigger|lower|more|fewer|same)', 0.82),
]


def _normalize(text: str) -> Optional[str]:
    """Normalize extracted text to standard comparison result."""
    text = text.lower().strip()
    for key, val in _NORMALIZE_MAP.items():
        if key in text:
            return val
    return None


class ComparisonStrategy:
    """Extract comparison results from model responses."""

    NAME = "comparison"

    def extract(self, text: str, expected_type: str = "comparison") -> ParseResult:
        if expected_type not in ("comparison", "auto"):
            return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

        # First check: "greater than", "less than", "equal to" as literal phrases
        for pattern, conf in _COMPARISON_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches:
                continue

            raw = matches[-1]
            if isinstance(raw, tuple):
                raw = raw[0]
            raw = raw.strip()

            # Check if this is a "Number 2 is X" pattern (needs inversion)
            idx = text.lower().rfind(raw.lower())
            surrounding = text[max(0, idx-50):idx+50].lower()
            needs_inversion = bool(re.search(
                r'(?:number\s*2|second number|num\s*2)\s+is\s+' + re.escape(raw.lower()),
                surrounding
            ))

            normalized = _normalize(raw)
            if normalized:
                if needs_inversion:
                    inv = {"greater than": "less than", "less than": "greater than", "equal to": "equal to"}
                    normalized = inv.get(normalized, normalized)
                return ParseResult(
                    value=normalized,
                    confidence=conf,
                    strategy_used=self.NAME,
                    raw_match=raw,
                )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)
