"""
Comprehensive parsing utilities for all beyondbench tasks.

This module provides robust parsing functions for extracting numerical answers
from model responses across all task types (easy, medium, hard).

Features:
- Multiple fallback strategies for answer extraction
- Support for LaTeX, Markdown, code blocks, and natural language
- Type-specific parsing for numbers, lists, booleans, grids
- Validation utilities for common answer formats
"""

import re
import json
import logging
import math
from typing import List, Optional, Union, Any, Tuple

# Re-export AnswerParser from parsing_utils for backwards compatibility
from .parsing_utils import AnswerParser


# ============================================================================
# Main Entry Points
# ============================================================================

def parse_boxed_answer(response: str, expected_type: str = "number") -> Optional[Union[int, float, str, List]]:
    """
    Extract answer from various boxed formats with type-specific handling.

    Args:
        response: The model's response text
        expected_type: Type of expected answer ("number", "list", "string", "boolean", "grid")

    Returns:
        Parsed answer if found, None otherwise
    """
    if not response:
        return None

    # Clean response for consistent processing
    clean_response = response.replace('\\n', ' ').strip()

    # Try multiple extraction methods in order of priority
    extracted_answer = (
        extract_from_boxed_format(clean_response) or
        extract_from_markdown_formatting(clean_response) or
        extract_from_final_answer(clean_response) or
        extract_from_explicit_statements(clean_response, expected_type) or
        extract_from_latex_math(clean_response) or
        extract_from_code_blocks(clean_response, expected_type) or
        extract_from_final_line(clean_response, expected_type) or
        extract_plain_value(clean_response, expected_type)
    )

    # Convert to expected type if found
    if extracted_answer is not None:
        return convert_to_type(extracted_answer, expected_type)

    return None


def extract_number(text: str) -> Optional[Union[int, float]]:
    """
    Standalone function to extract a single number from text.

    This is a convenience function for quick number extraction without
    the full parsing pipeline.

    Args:
        text: The text to extract a number from

    Returns:
        The extracted number (int or float), or None if not found
    """
    if not text:
        return None

    # Try boxed format first
    boxed = extract_from_boxed_format(text)
    if boxed:
        num = parse_number(boxed)
        if num is not None:
            return num

    # Direct number extraction
    return parse_number(text)


def extract_list(text: str) -> Optional[List[Union[int, float]]]:
    """
    Standalone function to extract a list of numbers from text.

    This is a convenience function for quick list extraction without
    the full parsing pipeline.

    Args:
        text: The text to extract a list from

    Returns:
        The extracted list of numbers, or None if not found
    """
    if not text:
        return None

    # Try boxed format first
    boxed = extract_from_boxed_format(text)
    if boxed:
        lst = parse_number_list(boxed)
        if lst:
            return lst

    # Direct list extraction
    return parse_number_list(text)


# ============================================================================
# Extraction Methods (in priority order)
# ============================================================================

def extract_from_boxed_format(text: str) -> Optional[str]:
    """Extract answer from various boxed formats.

    Handles a wide range of boxed format variations from different models
    including small models that may produce malformed LaTeX.
    """
    patterns = [
        # LaTeX boxed with text/textbf/mathrm commands (MUST come before simple boxed)
        r'\\boxed\{\\text\{([^}]+)\}\}',
        r'\\boxed\{\\textbf\{([^}]+)\}\}',
        r'\\boxed\{\\mathrm\{([^}]+)\}\}',
        r'\\boxed\{\\mathbf\{([^}]+)\}\}',
        # Standard LaTeX boxed format (handles negative numbers, lists, text)
        r'\\boxed\{([^{}]+)\}',
        # Nested braces handling (e.g., \boxed{-\frac{1}{2}})
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        # LaTeX boxed with math environment
        r'\\[\(\[]\\boxed\{([^{}]+)\}\\[\)\]]',
        # Dollar sign wrapped boxed
        r'\$\\boxed\{([^{}]+)\}\$',
        r'\$\$\\boxed\{([^{}]+)\}\$\$',
        # Markdown-style boxed
        r'\[boxed\{([^{}]+)\}\]',
        # LaTeX boxed with array command
        r'\\boxed\{\\begin\{array\}\{[^}]*\}([^}]+)\\end\{array\}\}',
        # Nested boxed
        r'\\boxed\{\\boxed\{([^{}]+)\}\}',
        # Simple boxed format without backslash
        r'(?<![\\a-zA-Z])boxed\{([^{}]+)\}',
        # Alternative boxed formats
        r'\\box\{([^{}]+)\}',
        r'\\fbox\{([^{}]+)\}',
        # Answer tags used by some models
        r'<answer>\s*([^<]+?)\s*</answer>',
        r'<solution>\s*([^<]+?)\s*</solution>',
        r'\[ANSWER\]\s*([^\[]+?)\s*\[/ANSWER\]',
        # Some small models use **answer** format at the end
        r'(?:final answer|the answer)(?:\s+is)?[:\s]+\*\*([^*]+)\*\*',
        # Handle boxed with escaped braces from some model outputs
        r'\\boxed\s*\{\s*([^{}]+?)\s*\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the last match (most likely the final answer)
            answer = matches[-1].strip()
            if not answer:
                continue
            # Clean LaTeX text commands while preserving content
            answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
            answer = re.sub(r'\\textbf\{([^}]*)\}', r'\1', answer)
            answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
            answer = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', answer)
            # Clean \displaystyle and similar commands
            answer = re.sub(r'\\displaystyle\s*', '', answer)
            # Clean remaining LaTeX artifacts (but preserve minus signs and commas)
            answer = re.sub(r'\\(?!-)[a-zA-Z]+\s*', '', answer)
            # Clean remaining braces
            answer = answer.replace('{', '').replace('}', '')
            # Normalize whitespace
            answer = re.sub(r'\s+', ' ', answer).strip()
            if answer:
                return answer

    return None


def extract_from_markdown_formatting(text: str) -> Optional[str]:
    """Extract answers from markdown bold/italic formatting.

    Only returns matches that look like actual answers (numbers, short phrases),
    not section headers or explanatory text in bold.
    """
    patterns = [
        # Bold with asterisks
        r'\*\*([^*]+)\*\*',
        # Bold with underscores
        r'__([^_]+)__',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Check if any match looks like an answer (contains numbers or is short)
            # Iterate from the end since the final answer is usually last
            for match in reversed(matches):
                match = match.strip()
                # Skip matches that look like headers or labels
                if match.endswith(':') or match.startswith('#'):
                    continue
                # Skip very long matches (likely explanatory text)
                if len(match) > 60:
                    continue
                # Accept if it's a number, a short phrase, or contains a number
                if re.search(r'-?\d', match) or len(match) < 30:
                    return match

    return None


def extract_from_final_answer(text: str) -> Optional[str]:
    """Extract answer from 'final answer' statements."""
    patterns = [
        # Final answer variations (case insensitive)
        r'(?:final answer|the final answer|my final answer)[:\s]+(?:is\s+)?(.+?)(?:\n|$|\.(?:\s|$))',
        r'(?:the answer|answer)[:\s]+(?:is\s+)?(.+?)(?:\n|$|\.(?:\s|$))',
        r'(?:the solution|solution)[:\s]+(?:is\s+)?(.+?)(?:\n|$|\.(?:\s|$))',
        r'(?:the result|result)[:\s]+(?:is\s+)?(.+?)(?:\n|$|\.(?:\s|$))',
        # Therefore/thus statements
        r'(?:therefore|thus|hence|so|consequently)[,\s]+(?:the )?(?:answer|result|solution) (?:is|are|would be|will be)[:\s]*(.+?)(?:\n|$|\.(?:\s|$))',
        # "X is the answer" format
        r'(.+?)\s+is the (?:final )?answer',
        r'(.+?)\s+is the (?:final )?result',
        r'(.+?)\s+is the (?:final )?solution',
        # Output format
        r'(?:output|the output)[:\s]+(?:is\s+)?(.+?)(?:\n|$|\.(?:\s|$))',
        # Common patterns from smaller models
        r'(?:the )?(?:answer|result|solution) is\s+(.+?)(?:\n|$|\.(?:\s|$))',
        # "= X" at end of line
        r'=\s*(.+?)(?:\n|$)',
        # "So the sorted list is: ..."
        r'(?:the )?sorted (?:list|array|sequence|order) (?:is|would be)[:\s]+(.+?)(?:\n|$|\.(?:\s|$))',
        # "The sum is X"
        r'(?:the )?(?:sum|total|count|product|difference|quotient|average|mean|median|mode|maximum|minimum|max|min|range) (?:is|equals?|would be|=)[:\s]*(.+?)(?:\n|$|\.(?:\s|$))',
        # "The next term is X" for sequence tasks
        r'(?:the )?next (?:term|number|element|value) (?:is|would be|=)[:\s]*(.+?)(?:\n|$|\.(?:\s|$))',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            result = matches[-1].strip()
            # Skip if the match is too long (likely explanation, not answer)
            if result and len(result) < 100:
                # Validate: must contain a digit, bracket, or boolean keyword
                if re.search(r'\d|[\[\(]|true|false|less than|greater than|equal to', result):
                    return result

    return None


def extract_from_explicit_statements(text: str, expected_type: str) -> Optional[str]:
    """Extract answer from explicit statements based on expected type."""
    if expected_type == "list":
        patterns = [
            r'(?:the |my |final )?(?:sorted|ordered|arranged|resulting) (?:list|array|sequence|numbers) (?:is|would be|will be|becomes)[:\s]+([^\n\.]+)',
            r'(?:after sorting|sorting gives|sorted version|sorted array)[:\s]+([^\n\.]+)',
            r'(?:the |my )?(?:list|array) (?:is|becomes)[:\s]+([^\n\.]+)',
        ]
    elif expected_type == "number":
        patterns = [
            r'(?:the |my |final )?(?:answer|result|value|number|count|sum|difference|product|quotient|total|average|mean|median|mode|maximum|minimum|max|min|length|output|score|index|range|variance|std|deviation) (?:is|would be|will be|becomes|equals)[:\s]+([^\n\.]+)',
            r'(?:has a |with a )?(?:length|count|value|size) (?:of|is|=)[:\s]+([^\n\.]+)',
            r'(?:equals|is equal to|amounts to|totals|sums to)[:\s]+([^\n\.]+)',
            r'(?:output|result)[:\s]+(\d+(?:\.\d+)?)',
            r'= ([+-]?\d+(?:\.\d+)?)',
        ]
    elif expected_type == "boolean":
        patterns = [
            r'(?:the |my )?(?:answer|result) (?:is|would be)[:\s]+(true|false|yes|no)',
            r'(true|false|yes|no)[,\s]*(?:is the answer)?',
        ]
    elif expected_type == "grid":
        patterns = [
            r'(?:the |my )?(?:grid|board|solution|puzzle) (?:is|would be)[:\s]+(.+)',
        ]
    else:
        patterns = [
            r'(?:the |my |final )?(?:answer|result|solution) (?:is|would be|will be|becomes)[:\s]+([^\n\.]+)',
        ]

    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            result = matches[-1].strip()
            # Skip matches that contain code block markers or look like garbage
            if result and not result.startswith('```') and re.search(r'\d', result):
                return result

    return None


def extract_from_latex_math(text: str) -> Optional[str]:
    """Extract answers from LaTeX math expressions."""
    patterns = [
        # Inline math
        r'\$([^$]+)\$',
        # Display math
        r'\$\$([^$]+)\$\$',
        # Parentheses math mode
        r'\\\(([^)]+)\\\)',
        # Brackets math mode
        r'\\\[([^\]]+)\\\]',
        # Equation environment
        r'\\begin{equation\*?}([^\\]+)\\end{equation\*?}',
        # Align environment
        r'\\begin{align\*?}([^\\]+)\\end{align\*?}',
        # Math environment
        r'\\begin{math}([^\\]+)\\end{math}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Clean LaTeX commands from the match
            answer = matches[-1].strip()
            answer = re.sub(r'\\[a-zA-Z]+\s*', '', answer)
            return answer

    return None


def extract_from_code_blocks(text: str, expected_type: str) -> Optional[str]:
    """Extract answer from code blocks."""
    # Find code blocks: ```X```, `X`
    patterns = [
        r'```(?:python|plaintext|text|output|result)?\s*\n?([^`]+)\n?```',
        r'`([^`]+)`',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                # Look for assignment or return statements
                code_patterns = [
                    r'(?:answer|result|solution|output)\s*=\s*([^;\n]+)',
                    r'return\s+([^;\n]+)',
                    r'print\s*\(\s*["\']?([^"\')\n]+)["\']?\s*\)',
                    r'>>> ([^\n]+)',  # Python REPL output
                    r'#\s*[Oo]utput[:\s]+(\S+)',  # # Output: 5
                    r'[Oo]utput[:\s]+(\d+(?:\.\d+)?)',  # Output: 5
                ]

                for code_pattern in code_patterns:
                    code_matches = re.findall(code_pattern, match)
                    if code_matches:
                        return code_matches[-1].strip()

                # If it's a simple expression (single line, single value), return it
                clean_match = match.strip()
                lines = [l.strip() for l in clean_match.split('\n') if l.strip()]
                if len(lines) == 1 and len(lines[0].split()) <= 3:
                    return lines[0]

    return None


def extract_from_final_line(text: str, expected_type: str) -> Optional[str]:
    """Extract answer from the final lines of text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if not lines:
        return None

    # Check the last few lines for answer patterns
    for i in range(min(8, len(lines))):
        line = lines[-(i+1)]

        # Skip lines that look like explanations (but be less aggressive)
        if len(line) > 200 or line.lower().startswith(('note:', 'explanation:', 'hint:')):
            continue

        # Look for standalone numbers or simple expressions
        if expected_type == "number":
            number_patterns = [
                r'^(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$',  # Just a number (with optional commas)
                r'^=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$',  # = number
                r'^:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$',  # : number
                r'(?:^|\s)(-?\d+(?:\.\d+)?)\s*[.\s]*$',  # Number at end of line
                # Handle "The answer is X." format
                r'(?:answer|result|sum|total|count|value)\s+(?:is|=)\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)',
                # Handle bold numbers: **42**
                r'\*\*(-?\d+(?:,\d{3})*(?:\.\d+)?)\*\*',
            ]
            for pattern in number_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).replace(',', '')

        elif expected_type == "list":
            list_patterns = [
                r'\[([^\[\]]+)\]',  # [a, b, c]
                r'\(([^()]+)\)',    # (a, b, c)
                r'\{([^{}]+)\}',    # {a, b, c}
                r'^(-?\d+(?:\s*,\s*-?\d+)+)\s*$',  # a, b, c (comma separated)
                r'^(-?\d+(?:\s+-?\d+)+)\s*$',  # a b c (space separated)
                # Handle bold lists: **[-3, 0, 5, 12]**
                r'\*\*\[([^\[\]]+)\]\*\*',
            ]
            for pattern in list_patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group(1)

        elif expected_type == "boolean":
            bool_patterns = [
                r'\b(true|false|yes|no)\b',
                r'\b(satisfiable|unsatisfiable)\b',
                r'\b(valid|invalid)\b',
                r'\b(possible|impossible)\b',
            ]
            for pattern in bool_patterns:
                match = re.search(pattern, line.lower())
                if match:
                    return match.group(1)

        elif expected_type == "string":
            # For string answers, look for quoted or emphasized text
            string_patterns = [
                r'"([^"]+)"',
                r"'([^']+)'",
                r'\*\*([^*]+)\*\*',
            ]
            for pattern in string_patterns:
                match = re.search(pattern, line)
                if match:
                    result = match.group(1).strip()
                    if result and len(result) < 50:
                        return result

    return None


def extract_plain_value(text: str, expected_type: str) -> Optional[str]:
    """Extract plain values when response is very short or simple.

    Also handles longer responses as a last resort by scanning for the
    most likely answer value in the text.
    """
    text = text.strip()

    # For very short responses, try direct extraction
    if len(text) < 100:
        if expected_type == "number":
            # Find any number in the text
            match = re.search(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
            if match:
                return match.group(1).replace(',', '')

        elif expected_type == "list":
            # Find list-like content
            match = re.search(r'[\[\(]([^\]\)]+)[\]\)]', text)
            if match:
                return match.group(1)
            # Also try comma-separated numbers without brackets
            match = re.search(r'(-?\d+(?:\s*,\s*-?\d+)+)', text)
            if match:
                return match.group(1)

        elif expected_type == "boolean":
            text_lower = text.lower()
            if 'true' in text_lower or 'yes' in text_lower:
                return 'true'
            elif 'false' in text_lower or 'no' in text_lower:
                return 'false'

        elif expected_type == "string":
            # For very short string responses, return the text itself
            if len(text) < 50:
                return text

    # For longer responses, try to find the answer near the end
    if len(text) >= 100:
        # Look at the last 500 characters for an answer
        tail = text[-500:]

        if expected_type == "number":
            # Find the last number in the response (most likely the final answer)
            numbers = re.findall(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)', tail)
            if numbers:
                return numbers[-1].replace(',', '')

        elif expected_type == "list":
            # Find the last list-like expression
            list_matches = re.findall(r'\[([^\[\]]+)\]', tail)
            if list_matches:
                return list_matches[-1]

    return None


# ============================================================================
# Type Conversion Functions
# ============================================================================

def convert_to_type(value: str, expected_type: str) -> Optional[Union[int, float, str, List, bool]]:
    """Convert extracted value to expected type."""
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

    if not value:
        return None

    try:
        if expected_type == "number":
            return parse_number(value)
        elif expected_type == "list":
            return parse_number_list(value)
        elif expected_type == "boolean":
            return parse_boolean(value)
        elif expected_type == "grid":
            return parse_grid(value)
        elif expected_type == "string":
            return str(value)
        else:
            # Try to intelligently parse
            if isinstance(value, str) and value.lower() in ['true', 'false', 'yes', 'no']:
                return parse_boolean(value)
            elif isinstance(value, str) and any(char in value for char in ['[', '(', ',']):
                parsed_list = parse_number_list(value)
                return parsed_list if parsed_list else value
            else:
                parsed_num = parse_number(value)
                return parsed_num if parsed_num is not None else value
    except Exception as e:
        logging.debug(f"Type conversion error: {e}")
        return None


def parse_number(text: Union[str, int, float]) -> Optional[Union[int, float]]:
    """
    Parse a single number from text with comprehensive handling.

    Handles:
    - Plain integers and floats
    - Numbers with thousand separators (1,000,000)
    - Scientific notation (1e10, 1.5E-3)
    - Fractions (1/2, 3/4)
    - Negative numbers
    - LaTeX formatted numbers
    """
    if text is None:
        return None

    # If already a number, return it
    if isinstance(text, (int, float)):
        if math.isnan(text) or math.isinf(text):
            return None
        return text

    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None

    # Clean the text
    text = text.strip()

    if not text:
        return None

    # Remove LaTeX formatting
    text = re.sub(r'\\[a-zA-Z]+\s*', '', text)
    text = re.sub(r'[{}$]', '', text)

    # Remove markdown formatting
    text = re.sub(r'[*_~`]', '', text)

    # Handle space between minus sign and number
    text = re.sub(r'-\s+(\d)', r'-\1', text)

    # Handle fractions first
    fraction_match = re.search(r'(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', text)
    if fraction_match:
        try:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator != 0:
                result = numerator / denominator
                return int(result) if result == int(result) else result
        except (ValueError, ZeroDivisionError):
            pass

    # Handle scientific notation
    sci_match = re.search(r'(-?\d+(?:\.\d+)?)\s*[eE]\s*([+-]?\d+)', text)
    if sci_match:
        try:
            mantissa = float(sci_match.group(1))
            exponent = int(sci_match.group(2))
            result = mantissa * (10 ** exponent)
            return int(result) if result == int(result) and abs(result) < 10**15 else result
        except (ValueError, OverflowError):
            pass

    # Remove thousand separators and extract number
    # Pattern matches numbers with optional thousand separators
    number_pattern = r'([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)'
    number_match = re.search(number_pattern, text)

    if number_match:
        num_str = number_match.group(1)
        num_str = num_str.replace(',', '')  # Remove thousand separators
        try:
            if '.' in num_str:
                return float(num_str)
            else:
                return int(num_str)
        except ValueError:
            pass

    # Fallback: try simple number extraction
    simple_match = re.search(r'(-?\d+(?:\.\d+)?)', text)
    if simple_match:
        try:
            num_str = simple_match.group(1)
            if '.' in num_str:
                return float(num_str)
            else:
                return int(num_str)
        except ValueError:
            pass

    return None


def parse_number_list(text: Union[str, List]) -> Optional[List[Union[int, float]]]:
    """
    Parse a list of numbers from text with comprehensive handling.

    Handles:
    - JSON arrays: [1, 2, 3]
    - Tuple notation: (1, 2, 3)
    - Set notation: {1, 2, 3}
    - Comma-separated: 1, 2, 3
    - Space-separated: 1 2 3
    - LaTeX arrays: \\begin{bmatrix} 1 \\\\ 2 \\\\ 3 \\end{bmatrix}
    - Python lists: list([1, 2, 3])
    """
    if text is None:
        return None

    # If already a list, try to convert elements
    if isinstance(text, list):
        try:
            numbers = []
            for x in text:
                num = parse_number(x) if not isinstance(x, (int, float)) else x
                if num is not None:
                    numbers.append(num)
            return numbers if numbers else None
        except:
            return None

    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None

    # Clean the text
    text = text.strip()

    if not text:
        return None

    # Try to parse as JSON first
    # Handle various bracket types and clean whitespace
    json_candidate = text.strip()
    if (json_candidate.startswith('[') and json_candidate.endswith(']')) or \
       (json_candidate.startswith('(') and json_candidate.endswith(')')):
        # Convert tuple notation to list notation for JSON parsing
        json_text = json_candidate.replace('(', '[').replace(')', ']')
        # Clean up common formatting issues from model outputs
        json_text = re.sub(r',\s*,', ',', json_text)  # Remove double commas
        json_text = re.sub(r',\s*\]', ']', json_text)  # Remove trailing comma
        json_text = re.sub(r'\[\s*,', '[', json_text)  # Remove leading comma
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, list):
                numbers = []
                for x in parsed:
                    num = parse_number(x)
                    if num is not None:
                        numbers.append(num)
                return numbers if numbers else None
        except json.JSONDecodeError:
            pass

    # Handle LaTeX array notations
    text = re.sub(r'\\\\', ',', text)  # Replace LaTeX line breaks
    text = re.sub(r'\\begin{[a-z]*matrix}', '', text)  # Remove matrix begin
    text = re.sub(r'\\end{[a-z]*matrix}', '', text)    # Remove matrix end
    text = re.sub(r'\\[a-zA-Z]+(?:{[^{}]*})?', '', text)  # Remove other LaTeX commands

    # Remove brackets, braces, parentheses
    text = re.sub(r'[\[\](){}]', '', text)

    # Handle "and" separators
    text = re.sub(r'\s+and\s+', ', ', text, flags=re.IGNORECASE)

    # Split by common separators and extract numbers
    numbers = []
    parts = re.split(r'[,;\s]+', text)

    for part in parts:
        part = part.strip()
        if part:
            num = parse_number(part)
            if num is not None:
                numbers.append(num)

    return numbers if numbers else None


def parse_boolean(text: Union[str, bool]) -> Optional[bool]:
    """
    Parse a boolean value from text.

    Handles various representations:
    - true/false
    - yes/no
    - 1/0
    - correct/incorrect
    - right/wrong
    - satisfied/unsatisfied
    """
    if text is None:
        return None

    if isinstance(text, bool):
        return text

    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None

    text = text.strip().lower()

    true_values = ['true', 'yes', '1', 'correct', 'right', 'satisfied', 'satisfiable',
                   'valid', 'possible', 'solvable', 'feasible', 't', 'y']
    false_values = ['false', 'no', '0', 'incorrect', 'wrong', 'unsatisfied', 'unsatisfiable',
                    'invalid', 'impossible', 'unsolvable', 'infeasible', 'f', 'n']

    if text in true_values:
        return True
    elif text in false_values:
        return False

    return None


def parse_grid(text: str) -> Optional[List[List[Union[int, str]]]]:
    """
    Parse a 2D grid from text.

    Handles:
    - Nested arrays: [[1,2,3],[4,5,6]]
    - Line-separated rows: "1 2 3\n4 5 6"
    - Pipe-separated: "1|2|3\n4|5|6"
    - Sudoku-style: "123\n456\n789"
    """
    if not text:
        return None

    text = text.strip()

    # Try JSON parsing first
    if text.startswith('['):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
                return parsed
        except json.JSONDecodeError:
            pass

    # Try line-by-line parsing
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if not lines:
        return None

    grid = []
    for line in lines:
        # Skip separator lines
        if re.match(r'^[-+|_=]+$', line):
            continue

        # Remove grid borders
        line = re.sub(r'^[|]', '', line)
        line = re.sub(r'[|]$', '', line)

        # Split by common separators
        if '|' in line:
            cells = line.split('|')
        elif ',' in line:
            cells = line.split(',')
        elif '\t' in line:
            cells = line.split('\t')
        else:
            # Try space separation or individual characters
            cells = line.split() if ' ' in line else list(line)

        row = []
        for cell in cells:
            cell = cell.strip()
            if cell:
                # Try to parse as number
                num = parse_number(cell)
                row.append(num if num is not None else cell)

        if row:
            grid.append(row)

    return grid if grid else None


# ============================================================================
# Specific Task Parsing Functions
# ============================================================================

def parse_sorted_list(response: str) -> Optional[List[int]]:
    """Parse a sorted list from model response."""
    result = parse_boxed_answer(response, "list")
    if isinstance(result, list):
        try:
            return [int(x) for x in result]
        except (ValueError, TypeError):
            pass
    return None


def parse_count(response: str) -> Optional[int]:
    """Parse a count (integer) from model response."""
    result = parse_boxed_answer(response, "number")
    if isinstance(result, (int, float)):
        return int(result)
    return None


def parse_sum(response: str) -> Optional[Union[int, float]]:
    """Parse a sum from model response."""
    return parse_boxed_answer(response, "number")


def parse_index(response: str) -> Optional[int]:
    """Parse an index (integer) from model response."""
    result = parse_boxed_answer(response, "number")
    if isinstance(result, (int, float)):
        return int(result)
    return None


def parse_comparison_result(response: str) -> Optional[Union[int, float]]:
    """Parse a comparison result from model response."""
    return parse_boxed_answer(response, "number")


def parse_arithmetic_result(response: str) -> Optional[Union[int, float]]:
    """Parse an arithmetic operation result from model response."""
    return parse_boxed_answer(response, "number")


def parse_statistical_result(response: str) -> Optional[Union[int, float]]:
    """Parse a statistical calculation result from model response."""
    return parse_boxed_answer(response, "number")


def parse_sequence_result(response: str) -> Optional[Union[int, float, List]]:
    """Parse a sequence-related result from model response."""
    # First try as a single number
    number_result = parse_boxed_answer(response, "number")
    if number_result is not None:
        return number_result

    # Then try as a list
    list_result = parse_boxed_answer(response, "list")
    if list_result is not None:
        return list_result

    return None


def parse_boolean_result(response: str) -> Optional[bool]:
    """Parse a boolean result from model response."""
    return parse_boxed_answer(response, "boolean")


def parse_grid_result(response: str) -> Optional[List[List]]:
    """Parse a grid/matrix result from model response."""
    return parse_boxed_answer(response, "grid")


# ============================================================================
# Robust Answer Extraction (combines all methods)
# ============================================================================

def robust_extract_answer(response: str, expected_type: str = "auto") -> Tuple[bool, Optional[Any]]:
    """
    Extract an answer from LLM response with maximum robustness.

    This function tries all extraction methods and returns both the
    extracted answer and whether the instruction was followed (boxed format).

    Args:
        response: The model's response text
        expected_type: Expected answer type ("number", "list", "boolean", "grid", "auto")

    Returns:
        Tuple of (instruction_followed, extracted_answer)
    """
    if not response:
        return False, None

    clean_response = response.replace('\\n', ' ').strip()

    # Try boxed format first (highest priority, instruction followed)
    boxed_answer = extract_from_boxed_format(clean_response)
    if boxed_answer:
        if expected_type == "auto":
            # Auto-detect type
            if any(c in boxed_answer for c in ['[', '(', ',']):
                result = parse_number_list(boxed_answer)
                if result:
                    return True, result
            if boxed_answer.lower() in ['true', 'false', 'yes', 'no']:
                return True, parse_boolean(boxed_answer)
            result = parse_number(boxed_answer)
            if result is not None:
                return True, result
            return True, boxed_answer
        else:
            result = convert_to_type(boxed_answer, expected_type)
            if result is not None:
                return True, result

    # Try other methods (instruction not followed)
    result = parse_boxed_answer(clean_response, expected_type if expected_type != "auto" else "number")
    if result is not None:
        return False, result

    return False, None


# ============================================================================
# Validation Functions
# ============================================================================

def validate_sorted_list(parsed_result: List, original_list: List) -> bool:
    """Validate that a parsed list is the correct sorted version of the original."""
    if not isinstance(parsed_result, list) or not isinstance(original_list, list):
        return False
    return sorted(original_list) == list(parsed_result)


def validate_count_result(parsed_result: int, expected_count: int) -> bool:
    """Validate that a parsed count matches the expected count."""
    return isinstance(parsed_result, int) and parsed_result == expected_count


def validate_arithmetic_result(parsed_result: Union[int, float], expected_result: Union[int, float],
                               tolerance: float = 1e-6) -> bool:
    """Validate that a parsed arithmetic result matches the expected result within tolerance."""
    if not isinstance(parsed_result, (int, float)) or not isinstance(expected_result, (int, float)):
        return False
    return abs(parsed_result - expected_result) <= tolerance


def validate_index_result(parsed_result: int, max_index: int) -> bool:
    """Validate that a parsed index is within valid bounds."""
    return isinstance(parsed_result, int) and 0 <= parsed_result < max_index


def validate_grid(parsed_grid: List[List], expected_grid: List[List]) -> bool:
    """Validate that a parsed grid matches the expected grid."""
    if not isinstance(parsed_grid, list) or not isinstance(expected_grid, list):
        return False
    if len(parsed_grid) != len(expected_grid):
        return False
    for row1, row2 in zip(parsed_grid, expected_grid):
        if not isinstance(row1, list) or not isinstance(row2, list):
            return False
        if len(row1) != len(row2):
            return False
        if row1 != row2:
            return False
    return True


# ============================================================================
# Utility Functions
# ============================================================================

def is_valid_number(value: Any) -> bool:
    """Check if a value is a valid number."""
    if value is None:
        return False
    if isinstance(value, bool):
        return False  # Booleans are technically ints in Python, but we don't want them
    if isinstance(value, (int, float)):
        return not (math.isnan(value) or math.isinf(value))
    if isinstance(value, str):
        try:
            float(value.replace(',', ''))
            return True
        except ValueError:
            return False
    return False


def clean_response(response: str) -> str:
    """Clean a response string for parsing."""
    if not response:
        return ""

    # Normalize newlines
    response = response.replace('\\n', '\n')

    # Remove excessive whitespace
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r' {2,}', ' ', response)

    return response.strip()


def extract_all_numbers(text: str) -> List[Union[int, float]]:
    """Extract all numbers from text in order of appearance."""
    if not text:
        return []

    numbers = []
    pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?)'

    for match in re.finditer(pattern, text):
        num_str = match.group(1).replace(',', '')
        try:
            if '.' in num_str:
                numbers.append(float(num_str))
            else:
                numbers.append(int(num_str))
        except ValueError:
            continue

    return numbers
