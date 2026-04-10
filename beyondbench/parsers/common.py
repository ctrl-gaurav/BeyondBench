"""
Shared extraction functions for all beyondbench parsers.

This module centralizes the common parsing logic that was previously duplicated
across 14 individual parser files. All parser files should import from here
instead of re-implementing extraction logic.

Functions:
    extract_from_boxed_formats: Extract answers from LaTeX boxed formats
    extract_from_explicit_statements: Extract answers from natural language statements
    clean_and_convert_to_number: Clean text and convert to numeric value
    extract_from_last_line: Extract answers from the last line/sentence
    extract_from_latex_math: Extract answers from LaTeX math expressions
    extract_from_code_blocks: Extract answers from code blocks
    extract_from_markdown_formatting: Extract answers from bold/italic text
    extract_input_numbers_from_prompt: Extract input numbers from prompt text
    is_valid_number: Check if a value is a valid number
    is_input_number: Check if a number is from the input prompt
"""

import re
import math
import logging
from typing import Optional, Union, Tuple, List

logger = logging.getLogger(__name__)


def extract_from_boxed_formats(text: str) -> Tuple[Optional[str], bool]:
    """
    Extract answers from various boxed formats.

    Handles:
    - Standard LaTeX: \\boxed{answer}
    - Nested braces: \\boxed{\\text{42}}, \\boxed{\\textbf{42}}
    - Double-dollar LaTeX: $$\\boxed{42}$$
    - Malformed boxes: \\boxed{42 (missing closing brace)
    - Multiple boxed answers: takes the LAST one
    - Alternative formats: \\box{}, \\fbox{}, <answer> tags

    Args:
        text: The response text to parse

    Returns:
        Tuple of (extracted_answer, instruction_followed_flag).
        instruction_followed is True when a boxed format was found.
    """
    if not text:
        return None, False

    # Patterns ordered from most specific to most general
    patterns = [
        # Nested LaTeX text commands inside boxed
        r'\\boxed\{\\text\{([^}]+)\}\}',
        r'\\boxed\{\\textbf\{([^}]+)\}\}',
        r'\\boxed\{\\mathrm\{([^}]+)\}\}',
        r'\\boxed\{\\mathbf\{([^}]+)\}\}',
        # Double-dollar wrapped boxed
        r'\$\$\s*\\boxed\{([^{}]+)\}\s*\$\$',
        # Single-dollar wrapped boxed
        r'\$\\boxed\{([^{}]+)\}\$',
        # LaTeX math environment with boxed
        r'\\[\(\[]\\boxed\{([^{}]+)\}\\[\)\]]',
        # Standard LaTeX boxed (handles nested braces for fractions etc.)
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        # Markdown-style boxed
        r'\[boxed\{([^{}]+)\}\]',
        # Alternative boxed formats
        r'\\box\{([^{}]+)\}',
        r'\\fbox\{([^{}]+)\}',
        # Answer tags used by some models
        r'<answer>\s*([^<]+?)\s*</answer>',
        r'<solution>\s*([^<]+?)\s*</solution>',
        r'\[ANSWER\]\s*([^\[]+?)\s*\[/ANSWER\]',
        # Bold final answer format
        r'(?:final answer|the answer)(?:\s+is)?[:\s]+\*\*([^*]+)\*\*',
    ]

    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            all_matches.extend(matches)

    if all_matches:
        # Take the last match (most likely the final answer)
        answer = all_matches[-1].strip()
        # Clean LaTeX text commands while preserving content
        answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\textbf\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\displaystyle\s*', '', answer)
        # Clean remaining LaTeX commands (preserve minus signs)
        answer = re.sub(r'\\(?!-)[a-zA-Z]+\s*', '', answer)
        # Clean remaining braces
        answer = answer.replace('{', '').replace('}', '')
        answer = re.sub(r'\s+', ' ', answer).strip()
        if answer:
            return answer, True

    # Handle malformed boxed: \boxed{42 (missing closing brace)
    malformed = re.findall(r'\\boxed\{([^{}]+)$', text, re.MULTILINE)
    if malformed:
        answer = malformed[-1].strip()
        answer = re.sub(r'\\[a-zA-Z]+\s*', '', answer)
        answer = answer.replace('{', '').replace('}', '')
        answer = re.sub(r'\s+', ' ', answer).strip()
        if answer:
            return answer, True

    return None, False


def extract_from_explicit_statements(text: str) -> Tuple[Optional[str], bool]:
    """
    Extract answers from explicit natural language statements.

    Handles:
    - "The answer is X", "Answer: X", "Final answer: X"
    - "Therefore, X" / "So, X" / "Hence, X" / "Thus, X"
    - "The answer is approximately X" (extracts X)
    - "X is the answer" (inverted pattern)
    - Numbered list answers: "1. 42" at end
    - Multilingual patterns (Chinese, Japanese, Korean)
    - "The sum/total/result/value is X"

    Args:
        text: The response text to parse

    Returns:
        Tuple of (extracted_answer, instruction_followed_flag).
    """
    if not text:
        return None, False

    patterns = [
        # "The answer is approximately X" — must come before general "answer is" patterns
        r'(?:the )?(?:answer|result|value)\s+is\s+(?:approximately|about|roughly|around)\s+([+-]?\d+(?:\.\d+)?)',
        # Standard answer patterns
        r'(?:^|\n|\s)(?:final answer|the final answer is|the final answer)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))',
        r'(?:^|\n|\s)(?:answer|the answer is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))',
        r'(?:^|\n|\s)(?:the result is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))',
        r'(?:^|\n|\s)(?:the value is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))',
        r'(?:^|\n|\s)(?:the number is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))',
        r'(?:^|\n|\s)(?:the output is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))',
        r'(?:^|\n|\s)(?:the solution is)[:\s]+(.+?)\s*(?:\n|$|\.(?!\d))',
        # Therefore/Thus/Hence/So patterns
        r'(?:therefore|thus|hence|so|consequently)[,\s]+(?:the )?(?:answer|result|value|solution|sum|total|count|product|difference|quotient)\s+(?:is|would be|=)\s+(.+?)\s*(?:\n|$|\.(?!\d))',
        # "The sum/total/result is X"
        r'(?:^|\n|\s)(?:the )?(?:total|sum|result|output|answer|value|number|count|product|difference|quotient|average|mean|median|mode|maximum|minimum|max|min|range)(?: of [^.]*)?\s+(?:is|=|equals)\s+(.+?)\s*(?:\n|$)',
        # Inverted patterns: "X is the answer"
        r'(?:^|\n|\s)(.+?)\s+is the (?:final )?answer\b',
        r'(?:^|\n|\s)(.+?)\s+is the (?:final )?result\b',
        r'(?:^|\n|\s)(.+?)\s+is the (?:final )?value\b',
        r'(?:^|\n|\s)(.+?)\s+is the (?:final )?solution\b',
        # Numbered list answer at end
        r'(?:^|\n)\d+[.)]\s*(.+?)\s*$',
        # Multilingual patterns (Chinese)
        r'(?:\u7b54\u6848|\u7ed3\u679c|\u603b\u548c)(?:\u662f|[:：])\s*(.+?)\s*(?:\n|$|[。\.])',
        # Multilingual patterns (Japanese)
        r'(?:\u7b54\u3048|\u7d50\u679c)(?:\u306f|[:：])\s*(.+?)\s*(?:\n|$|[。\.])',
        # Multilingual patterns (Korean)
        r'(?:\ub2f5|\uacb0\uacfc)(?:\ub294|\uc740|[:：])\s*(.+?)\s*(?:\n|$|[。\.])',
        # "= X" at end of line (common in math)
        r'=\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*$',
        # Last resort: standalone number on a line (common for terse outputs)
        r'(?:^|\n)\s*([-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:\n|$)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            match = matches[-1]
            if isinstance(match, tuple):
                match = match[0]
            cleaned_match = match.strip()
            if cleaned_match and len(cleaned_match) < 100:
                return cleaned_match, False

    return None, False


def clean_and_convert_to_number(text: str) -> Optional[Union[int, float, str]]:
    """
    Clean text and convert it to a number if possible.

    Handles:
    - Comma-separated thousands: "1,234,567" -> 1234567
    - Spaces in numbers: "1 234 567" -> 1234567
    - Scientific notation: "1.5e3" -> 1500.0
    - Fraction answers: "3/4" -> 0.75
    - "negative" word: "negative 5" -> -5
    - Parenthetical negatives: "(5)" -> -5
    - LaTeX/Markdown formatting removal
    - Trailing periods, colons, etc.

    Args:
        text: The text to clean and convert

    Returns:
        int, float, or the cleaned string if conversion fails. None if input is empty.
    """
    if not text:
        return None

    # Remove LaTeX formatting
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Remove markdown formatting
    text = re.sub(r'[*_~`]', '', text)
    # Remove trailing punctuation
    text = text.strip().rstrip('.,:;!?')

    # Handle "negative X" -> "-X"
    neg_match = re.match(r'^\s*negative\s+(\d+(?:\.\d+)?)\s*$', text, re.IGNORECASE)
    if neg_match:
        text = '-' + neg_match.group(1)

    # Handle "minus X" -> "-X"
    minus_match = re.match(r'^\s*minus\s+(\d+(?:\.\d+)?)\s*$', text, re.IGNORECASE)
    if minus_match:
        text = '-' + minus_match.group(1)

    # Handle parenthetical negatives: "(5)" -> "-5"
    paren_neg = re.match(r'^\s*\((\d+(?:\.\d+)?)\)\s*$', text)
    if paren_neg:
        text = '-' + paren_neg.group(1)

    # Handle fractions: "3/4" -> 0.75
    frac_match = re.match(r'^\s*(-?\d+)\s*/\s*(-?\d+)\s*$', text)
    if frac_match:
        numer = int(frac_match.group(1))
        denom = int(frac_match.group(2))
        if denom != 0:
            return numer / denom
        return None

    # Handle scientific notation: "1.5e3", "2E-4"
    sci_match = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*[eE]\s*([+-]?\d+)\s*$', text)
    if sci_match:
        try:
            return float(text.strip())
        except ValueError:
            pass

    # Handle spaces in numbers: "1 234 567" -> "1234567"
    space_num_match = re.match(r'^\s*([+-]?\d{1,3}(?:\s\d{3})+)\s*$', text)
    if space_num_match:
        text = space_num_match.group(1).replace(' ', '')

    # Standard number pattern: supports plain and comma-formatted numbers
    number_pattern = r'([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)'
    number_match = re.search(number_pattern, text)

    try:
        if number_match:
            number_str = number_match.group(1)
            number_str = number_str.replace(',', '')
            if '.' in number_str:
                return float(number_str)
            else:
                return int(number_str)
    except ValueError:
        pass

    # If still text, return cleaned text
    cleaned = text.strip()
    return cleaned if cleaned else None


def extract_from_last_line(text: str) -> Optional[str]:
    """
    Extract answers from the last line or sentence of the response.

    Args:
        text: The response text to parse

    Returns:
        Extracted answer string, or None
    """
    if not text:
        return None

    lines = text.strip().split('\n')

    number_pattern = r'([+-]?\d+(?:,\d{3})*)(?:\.\d+)?'
    quoted_text_pattern = r'[\"\']([^\"\'.]+)[\"\']'
    combined_pattern = fr'{quoted_text_pattern}\.?$|{number_pattern}\.?$'

    # Check the last few lines for potential answers
    for i in range(min(3, len(lines))):
        line = lines[-(i + 1)].strip()
        if line:
            answer_match = re.search(combined_pattern, line)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()

    # Split the text into sentences
    sentences = re.split(r'[.!?]', text)

    # Check the last few sentences for potential answers
    for i in range(min(3, len(sentences))):
        sentence = sentences[-(i + 1)].strip()
        if sentence:
            answer_match = re.search(combined_pattern, sentence)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()

    # Last resort: look for "= X" in last sentence
    if sentences:
        sentence = sentences[-1].split("=")
        match = re.search(number_pattern, sentence[-1])
        return match.group(0) if match else None

    return None


def extract_from_latex_math(text: str) -> Optional[str]:
    """
    Extract answers from LaTeX math expressions.

    Args:
        text: The response text to parse

    Returns:
        Extracted answer string, or None
    """
    if not text:
        return None

    patterns = [
        r'\$([^$]+)\$',
        r'\$\$([^$]+)\$\$',
        r'\\[\(\[]([^\\]+)\\[\)\]]',
        r'\\begin\{equation\}([^\\]+)\\end\{equation\}',
        r'\\begin\{align\}([^\\]+)\\end\{align\}',
        r'\\begin\{math\}([^\\]+)\\end\{math\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

    return None


def extract_from_code_blocks(text: str) -> Optional[str]:
    """
    Extract answers from code blocks.

    Args:
        text: The response text to parse

    Returns:
        Extracted answer string, or None
    """
    if not text:
        return None

    patterns = [
        r'```(?:python|plaintext)?\s*\n([^`]+)\n```',
        r'`([^`]+)`',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                print_match = re.search(r'print\s*\(\s*["\']?([^"\']+)["\']?\s*\)', match)
                if print_match:
                    return print_match.group(1).strip()

                return_match = re.search(r'return\s+([^\s;]+)', match)
                if return_match:
                    return return_match.group(1).strip()

                assign_match = re.search(r'(?:answer|result)\s*=\s*([^\s;]+)', match)
                if assign_match:
                    return assign_match.group(1).strip()

                # If the code block content is just a number, return it
                stripped = match.strip()
                if re.match(r'^[+-]?\d+(?:\.\d+)?$', stripped):
                    return stripped

    return None


def extract_from_markdown_formatting(text: str) -> Optional[str]:
    """
    Extract answers from markdown bold/italic formatting.

    Args:
        text: The response text to parse

    Returns:
        Extracted answer string, or None
    """
    if not text:
        return None

    patterns = [
        r'\*\*([^*]+)\*\*',
        r'__([^_]+)__',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                match = match.strip()
                if match.endswith(':') or match.startswith('#'):
                    continue
                if len(match) > 60:
                    continue
                if re.search(r'-?\d', match) or len(match) < 30:
                    return match

    return None


def extract_input_numbers_from_prompt(prompt: str) -> List[Union[int, float]]:
    """
    Extract input numbers from a prompt to avoid confusing them with answers.

    Args:
        prompt: The original prompt text

    Returns:
        List of numbers found in the prompt
    """
    if not prompt:
        return []

    numbers = []
    # Find all numbers in the prompt (including negatives and decimals)
    for match in re.finditer(r'(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)', prompt):
        try:
            val = float(match.group(1))
            if val == int(val):
                numbers.append(int(val))
            else:
                numbers.append(val)
        except ValueError:
            pass
    return numbers


def is_valid_number(value) -> bool:
    """
    Check if a value is a valid finite number.

    Args:
        value: The value to check

    Returns:
        True if the value is a valid finite number
    """
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
    if isinstance(value, str):
        try:
            f = float(value.replace(',', ''))
            return not (math.isnan(f) or math.isinf(f))
        except (ValueError, TypeError):
            return False
    return False


def is_input_number(value, input_numbers: List[Union[int, float]], tolerance: float = 1e-9) -> bool:
    """
    Check if a value matches any input number from the prompt.

    Args:
        value: The value to check
        input_numbers: List of numbers from the input prompt
        tolerance: Tolerance for float comparison

    Returns:
        True if the value matches an input number
    """
    if not input_numbers or value is None:
        return False

    try:
        num_val = float(value) if not isinstance(value, (int, float)) else value
        for inp in input_numbers:
            if abs(num_val - inp) < tolerance:
                return True
    except (ValueError, TypeError):
        pass

    return False
