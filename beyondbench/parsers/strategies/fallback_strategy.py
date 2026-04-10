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

    # Match integers, floats, comma-formatted, scientific notation, negative
    _NUM_RE = re.compile(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?', re.IGNORECASE)
    _STANDALONE_NUM_RE = re.compile(
        r'^[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?$', re.IGNORECASE
    )

    def extract(self, text: str, expected_type: str = "auto") -> ParseResult:
        if not text:
            return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # 1. Check last few non-empty lines for standalone numbers
        for line in reversed(lines[-5:]):
            if self._STANDALONE_NUM_RE.match(line):
                return ParseResult(
                    value=line.replace(',', ''),
                    confidence=0.55,
                    strategy_used=self.NAME + "_last_line_number",
                    raw_match=line,
                )

        # 2. Check for "**X**" bold answer pattern in last few lines
        for line in reversed(lines[-3:]):
            m = re.search(r'\*\*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', line)
            if m:
                return ParseResult(
                    value=m.group(1).replace(',', ''),
                    confidence=0.52,
                    strategy_used=self.NAME + "_bold_number",
                    raw_match=m.group(1),
                )

        # 3. Last number in the entire response
        all_nums = self._NUM_RE.findall(text)
        if all_nums:
            raw = all_nums[-1]
            return ParseResult(
                value=raw.replace(',', ''),
                confidence=0.40,
                strategy_used=self.NAME + "_last_number",
                raw_match=raw,
            )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)
