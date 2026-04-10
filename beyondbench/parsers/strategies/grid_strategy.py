"""
GridStrategy: Extract grid/matrix answers for Sudoku, N-Queens, etc.

Handles markdown tables, Python lists of lists, plain text rows.
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
    return all(len(row) == size for row in grid)


def _try_python_list(text: str) -> Optional[List[List]]:
    """Try to parse as Python list of lists."""
    # Find outermost list-of-lists pattern
    m = re.search(r'(\[\s*\[[\s\S]+?\]\s*\])', text)
    if not m:
        return None
    raw = m.group(1)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _try_markdown_table(text: str, size: int) -> Optional[List[List]]:
    """Try to parse markdown table rows."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('|--') or line.startswith('|-'):
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
    return None


def _try_plain_rows(text: str, size: int) -> Optional[List[List]]:
    """Try to parse plain text rows (space or comma separated)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are mostly words
        nums = re.findall(r'[+-]?\d+', line)
        if len(nums) == size:
            rows.append([int(n) for n in nums])
    if len(rows) >= size:
        # Take last `size` rows
        grid = rows[-size:]
        if _validate_grid(grid, size):
            return grid
    return None


class GridStrategy:
    """Extract grid/matrix answers from model responses."""

    NAME = "grid"

    # Common grid sizes (Sudoku 9x9, N-Queens 3-12, etc.)
    _COMMON_SIZES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def extract(self, text: str, expected_type: str = "grid") -> ParseResult:
        if expected_type not in ("grid", "auto"):
            return ParseResult(value=None, confidence=0.0, strategy_used=self.NAME)

        # Try Python list of lists first (highest confidence)
        grid = _try_python_list(text)
        if grid and isinstance(grid[0], list):
            size = len(grid[0])
            if _validate_grid(grid, size):
                return ParseResult(
                    value=str(grid),
                    confidence=0.90,
                    strategy_used=self.NAME + "_python_list",
                    raw_match=str(grid),
                )

        # Try each common size
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
