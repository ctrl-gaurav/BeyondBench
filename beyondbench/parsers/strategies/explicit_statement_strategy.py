"""
ExplicitStatementStrategy: Extract answers from natural language statements.

Handles: "The answer is X", "Therefore X", "X is the answer", multilingual patterns.
"""

import re
from typing import Optional
from .boxed_strategy import ParseResult

# (pattern, confidence, flags)
_EXPLICIT_PATTERNS = [
    # "The answer is approximately X" — before general "answer is"
    (r'(?:the )?(?:answer|result|value)\s+is\s+(?:approximately|about|roughly|around)\s+([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', 0.82, re.IGNORECASE),
    # Final answer patterns (highest priority)
    (r'(?:^|\n|\s)(?:final answer|the final answer)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))', 0.92, re.IGNORECASE | re.MULTILINE),
    # Standard answer patterns
    (r'(?:^|\n|\s)(?:answer|the answer)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))', 0.88, re.IGNORECASE | re.MULTILINE),
    (r'(?:^|\n|\s)(?:the result is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))', 0.85, re.IGNORECASE | re.MULTILINE),
    (r'(?:^|\n|\s)(?:the value is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))', 0.85, re.IGNORECASE | re.MULTILINE),
    (r'(?:^|\n|\s)(?:the output is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))', 0.83, re.IGNORECASE | re.MULTILINE),
    (r'(?:^|\n|\s)(?:the solution is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))', 0.83, re.IGNORECASE | re.MULTILINE),
    # Therefore/Thus/Hence patterns
    (r'(?:therefore|thus|hence|so|consequently)[,\s]+(?:the )?(?:answer|result|value|solution|sum|total|count|product|difference|quotient)\s+(?:is|would be|=)\s+(.+?)\s*(?:\n|$|\.(?!\d))', 0.85, re.IGNORECASE | re.MULTILINE),
    # Task-specific: sum/total/count/etc
    (r'(?:^|\n|\s)(?:the )?(?:total|sum|result|output|answer|value|number|count|product|difference|quotient|average|mean|median|mode|maximum|minimum|max|min|range)(?: of [^.]*)?\s+(?:is|=|equals)\s+(.+?)\s*(?:\n|$)', 0.80, re.IGNORECASE | re.MULTILINE),
    # Inverted: "X is the answer"
    (r'(?:^|\n|\s)(.+?)\s+is the (?:final )?answer\b', 0.78, re.IGNORECASE | re.MULTILINE),
    (r'(?:^|\n|\s)(.+?)\s+is the (?:final )?result\b', 0.75, re.IGNORECASE | re.MULTILINE),
    # "= X" at end of line
    (r'=\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*$', 0.72, re.MULTILINE),
    # Multilingual — Chinese
    (r'(?:\u7b54\u6848|\u7ed3\u679c|\u603b\u548c)(?:\u662f|[:：])\s*(.+?)\s*(?:\n|$|[。.])', 0.80, 0),
    # Multilingual — Japanese
    (r'(?:\u7b54\u3048|\u7d50\u679c)(?:\u306f|[:：])\s*(.+?)\s*(?:\n|$|[。.])', 0.80, 0),
    # Multilingual — Korean
    (r'(?:\ub2f5|\uacb0\uacfc)(?:\ub294|\uc740|[:：])\s*(.+?)\s*(?:\n|$|[。.])', 0.80, 0),
    # Numbered list answer at end
    (r'(?:^|\n)\d+[.)]\s*(.+?)\s*$', 0.60, re.MULTILINE),
    # Standalone number on a line
    (r'(?:^|\n)\s*([-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:\n|$)', 0.55, re.MULTILINE),
]


class ExplicitStatementStrategy:
    """Extract answers from explicit natural language statements."""

    NAME = "explicit_statement"

    def extract(self, text: str) -> ParseResult:
        best: Optional[ParseResult] = None

        for pattern, conf, flags in _EXPLICIT_PATTERNS:
            try:
                matches = re.findall(pattern, text, flags)
            except re.error:
                continue

            if not matches:
                continue

            raw = matches[-1]
            if isinstance(raw, tuple):
                raw = raw[0]
            cleaned = raw.strip()

            # Reject overly long matches (likely captured too much)
            if not cleaned or len(cleaned) > 200:
                continue

            result = ParseResult(
                value=cleaned,
                confidence=conf,
                strategy_used=self.NAME,
                raw_match=raw,
            )
            if best is None or conf > best.confidence:
                best = result

        return best or ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)
