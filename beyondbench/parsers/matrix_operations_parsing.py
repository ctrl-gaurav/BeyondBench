"""
Parser for matrix_operations task.

Handles determinant (scalar) and matrix answers in \\boxed{} or plain text.
"""

import re
import json
from typing import Optional, Union, List

from .common import extract_from_boxed_formats


def parse_matrix_answer(response: str, operation: str) -> Optional[Union[float, List]]:
    """
    Parse a matrix operations answer from a model response.

    For 'determinant': returns a float.
    For 'multiply' or 'inverse': returns a 2x2 list of lists of floats.
    """
    if not response:
        return None

    if operation == 'determinant':
        return _parse_scalar(response)
    else:
        return _parse_matrix(response)


def _parse_scalar(response: str) -> Optional[float]:
    """Extract a scalar number from response."""
    # Boxed format
    raw, found = extract_from_boxed_formats(response)
    if found and raw:
        try:
            return float(raw.strip())
        except (ValueError, TypeError):
            pass

    # All boxed occurrences
    for pattern in [r'\\boxed\{([^}]+)\}', r'\$\\boxed\{([^}]+)\}\$']:
        matches = re.findall(pattern, response)
        for raw in reversed(matches):
            try:
                return float(raw.strip())
            except (ValueError, TypeError):
                pass

    # Explicit statements
    patterns = [
        r'(?:determinant|det)[^\d-]*(-?\d+(?:\.\d+)?)',
        r'(?:answer|result|value)[^\d-]*(-?\d+(?:\.\d+)?)',
        r'(?:is|=)\s*(-?\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, response, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Last number
    nums = re.findall(r'-?\d+(?:\.\d+)?', response)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass

    return None


def _parse_matrix(response: str) -> Optional[List]:
    """Extract a 2x2 matrix from response."""
    # Try boxed matrix formats: \boxed{[[a,b],[c,d]]}
    # Handle both numeric and fraction entries
    boxed_patterns = [
        r'\\boxed\{\s*(\[\s*\[.+?\]\s*,\s*\[.+?\]\s*\])\s*\}',
        r'\\boxed\{\s*(\[.+?\])\s*\}',
    ]
    for pat in boxed_patterns:
        m = re.search(pat, response, re.DOTALL)
        if m:
            result = _try_parse_matrix_str(m.group(1))
            if result is not None:
                return result

    # Try JSON-like matrix in text (with fractions or decimals)
    num_pat = r'-?(?:\d+/\d+|\d+(?:\.\d+)?)'
    matrix_patterns = [
        rf'\[\s*\[\s*({num_pat})\s*,\s*({num_pat})\s*\]\s*,\s*\[\s*({num_pat})\s*,\s*({num_pat})\s*\]\s*\]',
    ]
    for pat in matrix_patterns:
        m = re.search(pat, response)
        if m:
            try:
                vals = []
                for i in range(1, 5):
                    v = m.group(i)
                    if '/' in v:
                        num, den = v.split('/')
                        vals.append(float(num) / float(den))
                    else:
                        vals.append(float(v))
                return [[vals[0], vals[1]], [vals[2], vals[3]]]
            except (ValueError, ZeroDivisionError):
                pass

    # Try LaTeX matrix formats: \begin{pmatrix}...\end{pmatrix}
    latex_mat = re.search(
        r'(?:pmatrix|bmatrix|matrix)\}?\s*'
        r'(-?[\d./]+)\s*&\s*(-?[\d./]+)\s*\\\\\s*(-?[\d./]+)\s*&\s*(-?[\d./]+)',
        response
    )
    if latex_mat:
        try:
            vals = []
            for i in range(1, 5):
                v = latex_mat.group(i)
                if '/' in v:
                    n, d = v.split('/')
                    vals.append(float(n) / float(d))
                else:
                    vals.append(float(v))
            return [[vals[0], vals[1]], [vals[2], vals[3]]]
        except (ValueError, ZeroDivisionError):
            pass

    # Try table-like rows with EXACTLY 2 numbers per line (but avoid lines from prompt context)
    # Only look in lines that appear AFTER "answer" or "result" keywords
    answer_start = max(
        response.lower().rfind('answer'),
        response.lower().rfind('result'),
        response.lower().rfind('therefore'),
        response.lower().rfind('so the'),
        0
    )
    search_region = response[answer_start:] if answer_start > 0 else response
    rows = []
    for line in search_region.split('\n'):
        # Match lines with exactly 2 numbers (possibly fractions)
        nums_raw = re.findall(r'-?(?:\d+/\d+|\d+(?:\.\d+)?)', line)
        if len(nums_raw) == 2:
            try:
                row_vals = []
                for v in nums_raw:
                    if '/' in v:
                        n, d = v.split('/')
                        row_vals.append(float(n) / float(d))
                    else:
                        row_vals.append(float(v))
                rows.append(row_vals)
            except (ValueError, ZeroDivisionError):
                pass
    if len(rows) == 2:
        return rows
    # If more than 2 rows, take the last 2 (most likely the answer)
    if len(rows) > 2:
        return rows[-2:]

    return None


def _try_parse_matrix_str(text: str) -> Optional[List]:
    """Try to parse a string like [[1,2],[3,4]] into a 2x2 list."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == 2:
            result = []
            for row in parsed:
                if isinstance(row, list) and len(row) == 2:
                    result.append([float(x) for x in row])
                else:
                    return None
            return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Manual parsing supporting fractions
    num_pat = r'-?(?:\d+/\d+|\d+(?:\.\d+)?)'
    m = re.match(
        rf'\[\s*\[\s*({num_pat})\s*,\s*({num_pat})\s*\]\s*,\s*\[\s*({num_pat})\s*,\s*({num_pat})\s*\]\s*\]',
        text
    )
    if m:
        try:
            vals = []
            for i in range(1, 5):
                v = m.group(i)
                if '/' in v:
                    num, den = v.split('/')
                    vals.append(float(num) / float(den))
                else:
                    vals.append(float(v))
            return [[vals[0], vals[1]], [vals[2], vals[3]]]
        except (ValueError, ZeroDivisionError):
            pass

    return None
