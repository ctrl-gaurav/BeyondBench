"""
Comparison result parser — delegates shared extraction to common.py.

Kept for backwards compatibility. New code should use UnifiedParser.
"""

import re
import logging

from .common import (
    extract_from_boxed_formats,
    extract_from_explicit_statements,
)


def parse_comparison_result(response):
    """
    Extract comparison result from a model response with robust pattern matching.

    Args:
        response (str): The model's response text

    Returns:
        str or None: Normalized comparison result ('greater than', 'less than', 'equal to')
                     if found, None otherwise
    """
    # Clean response for consistent processing
    clean_response = response.replace('\n', ' ').strip()

    # 1. Try boxed format via common.py
    boxed_answer, found = extract_from_boxed_formats(clean_response)
    if found and boxed_answer:
        normalized = normalize_comparison_result(boxed_answer)
        if normalized:
            return normalized

    # 2. Try explicit statements via common.py
    explicit_answer, _ = extract_from_explicit_statements(clean_response)
    if explicit_answer:
        normalized = normalize_comparison_result(explicit_answer)
        if normalized:
            return normalized

    # 3. Comparison-specific: final sentence scan
    result = extract_from_final_sentence(clean_response)
    if result:
        return normalize_comparison_result(result)

    # 4. Comparison-specific: symbolic comparison
    result = extract_from_comparison_symbols(clean_response)
    if result:
        return result

    return None


def extract_from_final_sentence(text):
    """Extract comparison result from the final sentence."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return None

    for i in range(min(2, len(sentences))):
        sentence = sentences[-(i + 1)].lower()

        if any(term in sentence for term in ['greater than', 'less than', 'equal to', '>', '<', '=']):
            for relation in ['greater than', 'less than', 'equal to']:
                if relation in sentence:
                    return relation
            if '>' in sentence:
                return 'greater than'
            elif '<' in sentence:
                return 'less than'
            elif '=' in sentence:
                return 'equal to'

    return None


def extract_from_comparison_symbols(text):
    """Extract comparison result from symbolic comparison statements."""
    match = re.search(r'(\d+)\s*([><]=?|=)\s*(\d+)', text)
    if match:
        symbol = match.group(2)
        num1 = int(match.group(1))
        num2 = int(match.group(3))

        num1_pattern = re.search(r'number\s*1\s*(?::|is|=)\s*(\d+)', text.lower())
        num2_pattern = re.search(r'number\s*2\s*(?::|is|=)\s*(\d+)', text.lower())

        if num1_pattern and num2_pattern:
            num1_value = int(num1_pattern.group(1))
            num2_value = int(num2_pattern.group(1))
            if num1 == num2_value and num2 == num1_value:
                symbol = '>' if symbol == '<' else ('<' if symbol == '>' else '=')

        if '>' in symbol:
            return 'greater than'
        elif '<' in symbol:
            return 'less than'
        elif symbol == '=':
            return 'equal to'

    return None


def normalize_comparison_result(result):
    """Normalize various comparison phrasings to standard format."""
    result = result.lower().strip()

    if any(term in result for term in ['greater', '>', 'more', 'larger', 'bigger', 'exceeds']):
        return 'greater than'
    elif any(term in result for term in ['less', '<', 'smaller', 'lower', 'below']):
        return 'less than'
    elif any(term in result for term in ['equal', '=', 'same', 'identical']):
        return 'equal to'

    # Handle "Number 1 is X than Number 2"
    if re.search(r'number\s*1.*number\s*2', result):
        if any(term in result for term in ['greater', '>', 'more', 'larger', 'bigger']):
            return 'greater than'
        elif any(term in result for term in ['less', '<', 'smaller', 'lower']):
            return 'less than'
        elif any(term in result for term in ['equal', '=', 'same']):
            return 'equal to'

    # Handle reversed: "Number 2 is X than Number 1"
    if re.search(r'number\s*2.*number\s*1', result):
        if any(term in result for term in ['greater', '>', 'more', 'larger', 'bigger']):
            return 'less than'
        elif any(term in result for term in ['less', '<', 'smaller', 'lower']):
            return 'greater than'
        elif any(term in result for term in ['equal', '=', 'same']):
            return 'equal to'

    return None
