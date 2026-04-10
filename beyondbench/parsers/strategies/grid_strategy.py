"""
GridStrategy: Extract grid/matrix answers for Sudoku, N-Queens, etc.

Handles:
- Python list of lists: [[1,2,3],[4,5,6],[7,8,9]]
- Markdown tables: | 1 | 2 | 3 |
- Plain text rows: "1 2 3 / 4 5 6" or one row per line
- Partial solutions: takes the last N rows where N matches expected size
"""

import re
import json
import ast
from typing import Optional, List
from .boxed_strategy import ParseResult


def _parse_row(text: str, size: int) -> Optional[List]:
    """Parse a single row of numbers."""
    nums = re.findall(r'[+-]?\d+', text)
    if len(nums) == size:
        return [int(n) for n in nums]
    return None


def _validate_grid(grid: List[List], size: int) -> bool:
    """Check grid has correct dimensions."""
    if len(grid) != size:
        return False
    return all(isinstance(row, list) and len(row) == size for row in grid)


def _try_python_list(text: str) -> Optional[List[List]]:
    """
    Try to parse a Python list-of-lists from text.

    Searches for the *last* occurrence of a list-of-lists pattern so that
    a model that re-states the puzzle before the answer doesn't confuse us.
    Also handles truncated outputs where the closing bracket may be missing.
    """
    # Find all outermost list-of-lists patterns
    candidates = re.findall(r'(\[\s*\[[\s\S]+?\]\s*\])', text)

    # Also try greedy match for potentially truncated outputs
    if not candidates:
        # Look for "[[ ... ]" without closing outer bracket
        truncated = re.findall(r'(\[\s*\[[\s\S]+\])\s*$', text)
        if truncated:
            raw = truncated[-1]
            if not raw.rstrip().endswith(']]'):
                raw = raw.rstrip() + ']'
            candidates = [raw]

    if not candidates:
        return None

    # Try from last to first (most likely to be the final answer)
    for raw in reversed(candidates):
        for loader in (ast.literal_eval, json.loads):
            try:
                parsed = loader(raw)
                if (isinstance(parsed, list) and parsed
                        and isinstance(parsed[0], list)):
                    return parsed
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    return None


def _try_markdown_table(text: str, size: int) -> Optional[List[List]]:
    """Try to parse markdown table rows."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or re.match(r'^\|[-:| ]+\|?$', line):
            continue
        if '|' in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            nums = []
            for c in cells:
                m = re.match(r'^[+-]?\d+$', c)
                if m:
                    nums.append(int(c))
            if len(nums) == size:
                rows.append(nums)
    if _validate_grid(rows, size):
        return rows
    # Partial: take last `size` rows
    if len(rows) > size:
        candidate = rows[-size:]
        if _validate_grid(candidate, size):
            return candidate
    return None


def _try_plain_rows(text: str, size: int) -> Optional[List[List]]:
    """
    Try to parse plain text rows (space/comma/slash separated).

    Also handles "1 2 3 / 4 5 6 / 7 8 9" on a single line.
    """
    # Expand slash-separated rows to multiple lines
    expanded = text.replace(' / ', '\n').replace('/', '\n')

    rows = []
    for line in expanded.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are mostly alphabetic (explanatory text)
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio > 0.4:
            continue
        nums = re.findall(r'[+-]?\d+', line)
        if len(nums) == size:
            rows.append([int(n) for n in nums])

    if _validate_grid(rows, size):
        return rows
    # Partial grid: take the last `size` rows
    if len(rows) > size:
        candidate = rows[-size:]
        if _validate_grid(candidate, size):
            return candidate
    return None


class GridStrategy:
    """Extract grid/matrix answers from model responses."""

    NAME = "grid"

    # Common grid sizes: Sudoku (4, 6, 9), N-Queens (3–12), plus 2 for edge cases
    _COMMON_SIZES = [4, 9, 6, 5, 8, 7, 3, 2, 10, 11, 12]

    def extract(self, text: str, expected_type: str = "grid") -> ParseResult:
        if expected_type not in ("grid", "auto"):
            return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

        # --- Attempt 1: Python list of lists (highest confidence) ---
        grid = _try_python_list(text)
        if grid and isinstance(grid[0], list):
            size = len(grid)
            if _validate_grid(grid, size):
                return ParseResult(
                    value=str(grid),
                    confidence=0.92,
                    strategy_used=self.NAME + "_python_list",
                    raw_match=str(grid),
                )

        # --- Attempts 2 & 3: markdown table and plain rows, for each size ---
        for size in self._COMMON_SIZES:
            grid = _try_markdown_table(text, size)
            if grid:
                return ParseResult(
                    value=str(grid),
                    confidence=0.85,
                    strategy_used=self.NAME + "_markdown",
                    raw_match=None,
                )

            grid = _try_plain_rows(text, size)
            if grid:
                return ParseResult(
                    value=str(grid),
                    confidence=0.78,
                    strategy_used=self.NAME + "_plain",
                    raw_match=None,
                )

        return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)
