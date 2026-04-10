"""
Sorted list parser — delegates shared extraction to common.py.

Kept for backwards compatibility. New code should use UnifiedParser.
"""

import re
import json
import logging

from .common import (
    extract_from_boxed_formats,
    extract_from_explicit_statements,
    extract_from_code_blocks,
    extract_from_last_line,
)


def parse_sorted_list(response):
    """
    Extract and parse a sorted list from a model response with robust pattern matching.

    Args:
        response (str): The model's response text

    Returns:
        list or None: Parsed list of integers if found, None otherwise
    """
    clean_response = response.replace('\\n', ' ').strip()

    # 1. Try boxed format via common.py
    boxed_answer, found = extract_from_boxed_formats(clean_response)
    if found and boxed_answer:
        parsed = parse_number_list(boxed_answer)
        if parsed:
            return parsed

    # 2. Try list-specific bracket formats
    result = extract_from_list_formats(clean_response)
    if result:
        return result

    # 3. Try sorting-specific explicit statements (before generic ones)
    result = extract_from_sorting_statements(clean_response)
    if result:
        return result

    # 4. Try code blocks via common.py
    code_answer = extract_from_code_blocks(clean_response)
    if code_answer:
        parsed = parse_number_list(code_answer)
        if parsed:
            return parsed

    # 5. Try generic explicit statements via common.py
    explicit_answer, _ = extract_from_explicit_statements(clean_response)
    if explicit_answer:
        parsed = parse_number_list(explicit_answer)
        if parsed:
            return parsed

    # 6. Try final line via common.py
    last_answer = extract_from_last_line(clean_response)
    if last_answer:
        parsed = parse_number_list(last_answer)
        if parsed:
            return parsed

    return None


def extract_from_sorting_statements(text):
    """Extract sorted list from sorting-specific explicit statements."""
    patterns = [
        r'(?:the |my |final )?(?:sorted|ordered|arranged|resulting) (?:list|array|sequence|numbers) (?:is|would be|will be|becomes)[:\s]+([^\.]+)',
        r'(?:after sorting|sorting gives|sorted version)[:\s]+([^\.]+)',
        r'(?:therefore|thus|hence|so|consequently)[,\s]+(?:the )?(?:sorted|ordered|arranged) (?:list|array|sequence|numbers) (?:is|are|would be|will be)[:\s]+([^\.]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in reversed(matches):
                cleaned_list = parse_number_list(match)
                if cleaned_list:
                    return cleaned_list

    return None


def extract_from_list_formats(text):
    """Extract sorted list from various list notation formats."""
    patterns = [
        r'\[([^\[\]]+)\]',
        r'\(([^()]+)\)',
        r'\\{([^{}]+)\\}',
        r'\\begin{[a-z]*matrix}([^}]+)\\end{[a-z]*matrix}',
        r'(?:list|sorted list|sorted array)[:=]\s*(?:\[|\{|\()([^]})]+)(?:\]|\}|\))',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                cleaned_list = parse_number_list(match)
                if cleaned_list:
                    return cleaned_list

    return None


def parse_number_list(text):
    """
    Parse a list of numbers from various text formats.

    Args:
        text (str or list): Text or list containing numbers

    Returns:
        list: List of integers if parsing successful, None otherwise
    """
    if not text:
        return None

    if isinstance(text, list):
        try:
            return [int(float(x)) if isinstance(x, (str, int, float)) else None for x in text]
        except (ValueError, TypeError):
            pass

    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return None

    text = text.strip()

    # Try JSON parse
    if (text.startswith('[') and text.endswith(']')) or (text.startswith('{') and text.endswith('}')):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                try:
                    return [int(float(x)) if isinstance(x, (str, int, float)) else None for x in parsed]
                except (ValueError, TypeError):
                    pass
        except json.JSONDecodeError:
            pass

    # Handle LaTeX formatting
    text = re.sub(r'\\\\', ',', text)
    text = re.sub(r'\\[a-zA-Z]+(?:{[^{}]*})?', '', text)

    # Split and parse numbers
    numbers = []
    try:
        parts = re.split(r'[,\s]+', text)
        for part in parts:
            part = part.strip()
            if part:
                part = part.replace(',', '')
                num = int(float(part))
                numbers.append(num)
        if numbers:
            return numbers
    except (ValueError, TypeError):
        pass

    # Fallback: extract any numbers in order
    try:
        numbers = [int(float(x)) for x in re.findall(r'-?\d+(?:\.\d+)?', text)]
        if numbers:
            return numbers
    except (ValueError, TypeError):
        pass

    return None
