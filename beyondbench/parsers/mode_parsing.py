"""
Mode parser — delegates shared extraction to common.py.

Kept for backwards compatibility. New code should use UnifiedParser.
"""

import re
import logging
from collections import Counter

from .common import (
    extract_from_boxed_formats,
    extract_from_explicit_statements,
    extract_from_latex_math,
    extract_from_code_blocks,
    extract_from_markdown_formatting,
    extract_from_last_line,
)


def parse_mode_answer(response):
    """
    Extract mode value(s) from model response with robust pattern matching.

    Args:
        response (str): The response from the LLM

    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    clean_response = response.replace('\n', ' ')

    # 1. Try boxed format via common.py
    boxed_answer, found = extract_from_boxed_formats(clean_response)
    if found and boxed_answer:
        modes = parse_modes_from_text(boxed_answer)
        if modes:
            return modes

    # 2. Try markdown formatting via common.py
    markdown_answer = extract_from_markdown_formatting(clean_response)
    if markdown_answer:
        modes = parse_modes_from_text(markdown_answer)
        if modes:
            return modes

    # 3. Try mode-specific explicit statements (before generic ones)
    mode_specific = extract_from_mode_statements(clean_response)
    if mode_specific:
        return mode_specific

    # 4. Try generic explicit statements via common.py
    explicit_answer, _ = extract_from_explicit_statements(clean_response)
    if explicit_answer:
        modes = parse_modes_from_text(explicit_answer)
        if modes:
            return modes

    # 5. Try LaTeX math via common.py
    latex_answer = extract_from_latex_math(clean_response)
    if latex_answer:
        modes = parse_modes_from_text(latex_answer)
        if modes:
            return modes

    # 5. Try code blocks via common.py
    code_answer = extract_from_code_blocks(clean_response)
    if code_answer:
        modes = parse_modes_from_text(code_answer)
        if modes:
            return modes

    # 7. Try array/list formats (mode-specific)
    array_modes = extract_from_array_format(clean_response)
    if array_modes:
        return array_modes

    # 8. Try last line via common.py
    last_answer = extract_from_last_line(clean_response)
    if last_answer:
        modes = parse_modes_from_text(last_answer)
        if modes:
            return modes

    return None


def extract_from_mode_statements(text):
    """Extract mode values from mode-specific explicit statements."""
    patterns = [
        r'(?:^|\n|\s)(?:mode|the mode is|modes are|the modes are)[:\s]+([^\n\.]+)',
        r'(?:^|\n|\s)(?:therefore,? the modes? (?:is|are))[:\s]+([^\n\.]+)',
        r'(?:^|\n|\s)(?:the most frequent values? (?:is|are))[:\s]+([^\n\.]+)',
        r'(?:^|\n|\s)(?:the values? that appears? most frequently (?:is|are))[:\s]+([^\n\.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            modes_text = match.group(1).strip()
            result = parse_modes_from_text(modes_text)
            if result:
                return result

    return None


def extract_from_array_format(text):
    """Extract mode values from array or list formats."""
    patterns = [
        r'\[\s*([^\[\]]*)\s*\]',
        r'\(\s*([^\(\)]*)\s*\)',
        r'\{\s*([^\{\}]*)\s*\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            modes_text = match.group(1).strip()
            return parse_modes_from_text(modes_text)

    return None


def parse_modes_from_text(text):
    """
    Parse a string containing comma-separated mode values into a list of integers.

    Args:
        text (str): The text containing comma-separated mode values

    Returns:
        list: The extracted mode values as a list of integers, or empty list if parsing fails
    """
    if not text:
        return []

    # Handle "X, Y, and Z" format
    text = re.sub(r'\s+and\s+', ', ', text)
    text = re.sub(r'\s+or\s+', ', ', text)
    text = re.sub(r'(\d+)\s+and\s+(\d+)', r'\1, \2', text)
    text = re.sub(r'(\d+)\s+or\s+(\d+)', r'\1, \2', text)
    text = re.sub(r'[;|]', ',', text)

    items = []
    for item in text.split(','):
        num_match = re.search(r'(-?\d+)', item)
        if num_match:
            try:
                items.append(int(num_match.group(1)))
            except ValueError:
                pass

    # Remove duplicates while preserving order
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return unique_items if unique_items else []


def extract_modes_from_frequency_count(text):
    """Extract modes by analyzing frequency counts in the response."""
    freq_pattern = r'(\d+)(?:\s*:\s*|\s+appears\s+)(\d+)(?:\s*times|\s*occurrences)'
    frequencies = re.findall(freq_pattern, text)

    if frequencies:
        value_freqs = [(int(value), int(freq)) for value, freq in frequencies]
        max_freq = max(freq for _, freq in value_freqs)
        modes = sorted([value for value, freq in value_freqs if freq == max_freq])
        return modes

    return None


def extract_from_table_format(text):
    """Extract modes from table formats in the response."""
    table_rows = re.findall(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|', text)

    if table_rows:
        value_freqs = [(int(value), int(freq)) for value, freq in table_rows]
        max_freq = max(freq for _, freq in value_freqs)
        modes = sorted([value for value, freq in value_freqs if freq == max_freq])
        return modes

    return None
