"""Parser for find minimum task answers."""

from .common import (
    extract_from_boxed_formats,
    extract_from_explicit_statements,
    extract_from_latex_math,
    extract_from_code_blocks,
    extract_from_last_line,
    clean_and_convert_to_number,
)


def parse_find_minimum_answer(clean_response):
    """
    Extract an answer from the LLM response for the find minimum task.

    Args:
        clean_response: The cleaned response from the LLM

    Returns:
        tuple: (instruction_followed, answer)
    """
    # 1. Try to extract from boxed formats (highest priority)
    boxed_answer, instruction_followed = extract_from_boxed_formats(clean_response)
    if boxed_answer is not None:
        answer = clean_and_convert_to_number(boxed_answer)
        return instruction_followed, answer

    # 2. Try to extract from explicit answer statements
    explicit_answer, _ = extract_from_explicit_statements(clean_response)
    if explicit_answer is not None:
        answer = clean_and_convert_to_number(explicit_answer)
        return False, answer

    # 3. Try to extract from LaTeX math expressions
    latex_answer = extract_from_latex_math(clean_response)
    if latex_answer is not None:
        answer = clean_and_convert_to_number(latex_answer)
        return False, answer

    # 4. Try to extract from code blocks
    code_answer = extract_from_code_blocks(clean_response)
    if code_answer is not None:
        answer = clean_and_convert_to_number(code_answer)
        return False, answer

    # 5. Try to extract from the last line or sentence
    last_line_answer = extract_from_last_line(clean_response)
    if last_line_answer is not None:
        answer = clean_and_convert_to_number(last_line_answer)
        return False, answer

    return False, None
