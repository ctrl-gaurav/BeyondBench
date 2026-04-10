"""Parser for division task answers."""

from .common import (
    extract_from_boxed_formats,
    extract_from_explicit_statements,
    extract_from_markdown_formatting,
    extract_from_latex_math,
    extract_from_code_blocks,
    extract_from_last_line,
    clean_and_convert_to_number,
    extract_input_numbers_from_prompt,
    is_valid_number,
    is_input_number,
)


def parse_division_answer(clean_response, prompt=None):
    """
    Extract an answer from the LLM response for the division task.

    Filters out input numbers from the prompt to avoid false positives.

    Args:
        clean_response: The cleaned response from the LLM
        prompt: The original prompt (used to filter input numbers)

    Returns:
        tuple: (instruction_followed, answer)
    """
    input_numbers = []
    if prompt:
        input_numbers = extract_input_numbers_from_prompt(prompt)

    # 1. Try to extract from boxed formats (highest priority)
    boxed_answer, instruction_followed = extract_from_boxed_formats(clean_response)
    if boxed_answer is not None:
        answer = clean_and_convert_to_number(boxed_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return instruction_followed, answer

    # 2. Try to extract from markdown formatting (bold, italic)
    markdown_answer = extract_from_markdown_formatting(clean_response)
    if markdown_answer is not None:
        answer = clean_and_convert_to_number(markdown_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return False, answer

    # 3. Try to extract from explicit answer statements
    explicit_answer, _ = extract_from_explicit_statements(clean_response)
    if explicit_answer is not None:
        answer = clean_and_convert_to_number(explicit_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return False, answer

    # 4. Try to extract from LaTeX math expressions
    latex_answer = extract_from_latex_math(clean_response)
    if latex_answer is not None:
        answer = clean_and_convert_to_number(latex_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return False, answer

    # 5. Try to extract from code blocks
    code_answer = extract_from_code_blocks(clean_response)
    if code_answer is not None:
        answer = clean_and_convert_to_number(code_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return False, answer

    # 6. Try to extract from the last line or sentence
    last_line_answer = extract_from_last_line(clean_response)
    if last_line_answer is not None:
        answer = clean_and_convert_to_number(last_line_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return False, answer

    return False, None
