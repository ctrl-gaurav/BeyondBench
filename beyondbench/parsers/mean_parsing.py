"""Parser for mean task answers."""

from .common import (
    extract_from_boxed_formats,
    extract_from_explicit_statements,
    extract_from_latex_math,
    extract_from_last_line,
    clean_and_convert_to_number,
    is_valid_number,
)


def parse_mean_answer(response):
    """
    Extract mean value from model response.

    Args:
        response: The model's response text

    Returns:
        float or None: The parsed mean value or None
    """
    # 1. Try to extract from boxed formats (highest priority)
    boxed_answer, _ = extract_from_boxed_formats(response)
    if boxed_answer is not None:
        answer = clean_and_convert_to_number(boxed_answer)
        if is_valid_number(answer):
            return answer

    # 2. Try to extract from explicit statements
    explicit_answer, _ = extract_from_explicit_statements(response)
    if explicit_answer is not None:
        answer = clean_and_convert_to_number(explicit_answer)
        if is_valid_number(answer):
            return answer

    # 3. Try to extract from LaTeX math expressions
    latex_answer = extract_from_latex_math(response)
    if latex_answer is not None:
        answer = clean_and_convert_to_number(latex_answer)
        if is_valid_number(answer):
            return answer

    # 4. Try to extract from the last line or sentence
    last_line_answer = extract_from_last_line(response)
    if last_line_answer is not None:
        answer = clean_and_convert_to_number(last_line_answer)
        if is_valid_number(answer):
            return answer

    return None
