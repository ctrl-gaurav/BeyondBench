"""
ListStrategy: Extract lists of numbers from model responses.

Handles comma-separated, bracket-enclosed, and newline-separated lists.
"""

import re
import json
from typing import Optional, List
from .boxed_strategy import ParseResult


def _parse_number_list(text: str) -> Optional[List]:
    """Try to parse text as a list of numbers."""
    if not text:
        return None

    text = text.strip()

    # Try JSON parse first if it looks like a list
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                result = []
                for x in parsed:
                    try:
                        f = float(str(x))
                        result.append(int(f) if f == int(f) else f)
                    except (ValueError, TypeError):
                        result.append(str(x))
                return result if result else None
        except json.JSONDecodeError:
            pass

    # Remove LaTeX formatting
    text = re.sub(r'\\[a-zA-Z]+(?:\{[^{}]*\})?', '', text)
    text = re.sub(r'\\\\', ',', text)  # LaTeX newlines → commas

    # Extract numbers
    numbers = []
    parts = re.split(r'[,;\s]+', text)
    for part in parts:
        part = part.strip().strip('[](){}')
        if not part:
            continue
        try:
            f = float(part.replace(',', ''))
            numbers.append(int(f) if f == int(f) else f)
        except ValueError:
            # Try extracting number from part
            m = re.search(r'[+-]?\d+(?:\.\d+)?', part)
            if m:
                try:
                    f = float(m.group())
                    numbers.append(int(f) if f == int(f) else f)
                except ValueError:
                    pass

    return numbers if len(numbers) >= 2 else None


class ListStrategy:
    """Extract lists of numbers from model responses."""

    NAME = "list"

    _BRACKET_PATTERNS = [
        # Standard array notation [a, b, c]
        (r'\[([^\[\]]+)\]', 0.88),
        # Parentheses notation (a, b, c) with at least one comma
        (r'\(([^()]+,[^()]+)\)', 0.80),
        # LaTeX set braces
        (r'\\{([^{}]+)\\}', 0.75),
        # LaTeX matrix environments
        (r'\\begin\{[a-z]*matrix\}([\s\S]+?)\\end\{[a-z]*matrix\}', 0.75),
        # "list: a, b, c" or "sorted list: a, b, c"
        (r'(?:list|sorted list|sorted array|sequence)[:\s]+([^.\n]+)', 0.72),
    ]

    def extract(self, text: str, expected_type: str = "list") -> ParseResult:
        if expected_type not in ("list", "auto"):
            return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

        all_candidates = []

        for pattern, conf in self._BRACKET_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for raw in matches:
                nums = _parse_number_list(raw)
                if nums and len(nums) >= 2:
                    all_candidates.append((nums, conf, raw))

        if all_candidates:
            # Take the last match
            nums, conf, raw = all_candidates[-1]
            return ParseResult(
                value=str(nums),
                confidence=conf,
                strategy_used=self.NAME,
                raw_match=raw,
            )

        # Fallback: look for comma-separated numbers anywhere
        m = re.search(r'(-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?){2,})', text)
        if m:
            raw = m.group(1)
            nums = _parse_number_list(raw)
            if nums:
                return ParseResult(
                    value=str(nums),
                    confidence=0.65,
                    strategy_used=self.NAME + "_csv",
                    raw_match=raw,
                )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)
