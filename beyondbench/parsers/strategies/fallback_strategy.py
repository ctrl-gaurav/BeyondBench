"""
FallbackStrategy: Last-resort number/value extraction.

Tries: last number on last line, last number in response, last line value.
"""

import re
from typing import Optional
from .boxed_strategy import ParseResult


class FallbackStrategy:
    """Last-resort extraction when all other strategies fail."""

    NAME = "fallback"

    def extract(self, text: str, expected_type: str = "auto") -> ParseResult:
        if not text:
            return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # 1. Check last few non-empty lines for standalone numbers
        for line in reversed(lines[-3:]):
            m = re.fullmatch(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?)', line)
            if m:
                return ParseResult(
                    value=m.group(1).replace(',', ''),
                    confidence=0.55,
                    strategy_used=self.NAME + "_last_line_number",
                    raw_match=line,
                )

        # 2. Last number in the entire response
        all_nums = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?', text)
        if all_nums:
            raw = all_nums[-1]
            return ParseResult(
                value=raw.replace(',', ''),
                confidence=0.40,
                strategy_used=self.NAME + "_last_number",
                raw_match=raw,
            )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)
