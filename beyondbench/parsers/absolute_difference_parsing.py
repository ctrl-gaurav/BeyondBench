import re

def parse_absolute_difference_answer(clean_response, prompt=None):
    """
    Extract an answer from the LLM response with robust pattern matching.
    
    This function attempts to extract the answer from various formats that LLMs might use,
    prioritizing the requested format (boxed) and falling back to other common patterns.
    It also avoids extracting numbers from the input list if prompt is provided.
    
    Args:
        clean_response (str): The cleaned response from the LLM
        prompt (str, optional): The prompt that was given to the LLM
        
    Returns:
        tuple: (bool, value) where bool indicates if instruction was followed
               and value is the extracted answer or None
    """
    # Extract input numbers from prompt if available
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
    explicit_answer, instruction_followed = extract_from_explicit_statements(clean_response)
    if explicit_answer is not None:
        answer = clean_and_convert_to_number(explicit_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return instruction_followed, answer
    
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
    
    # 7. Try to extract from scientific notation
    scientific_notation = extract_from_scientific_notation(clean_response)
    if scientific_notation is not None:
        answer = clean_and_convert_to_number(scientific_notation)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return False, answer
    
    # 8. Try to extract from array or list formats
    array_answer = extract_from_array_format(clean_response)
    if array_answer is not None:
        answer = clean_and_convert_to_number(array_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return False, answer
    
    # 9. Try to extract from calculation steps
    calculation_answer = extract_from_calculation_steps(clean_response, input_numbers)
    if calculation_answer is not None:
        answer = clean_and_convert_to_number(calculation_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            return False, answer
    
    # 10. Try to extract plain numbers from very short responses (lowest priority)
    # Only use this if we couldn't find any other answer and the response is very short
    if len(clean_response.strip()) < 10:
        plain_number = extract_plain_number(clean_response)
        if plain_number is not None and is_valid_number(plain_number) and not is_input_number(plain_number, input_numbers):
            return False, plain_number
    
    # 12. No valid answer found
    return False, None


def is_valid_number(value):
    """
    Check if a value is a valid number.
    
    Args:
        value: The value to check
        
    Returns:
        bool: True if the value is a valid number, False otherwise
    """
    if value is None:
        return False
    
    if isinstance(value, (int, float)):
        return True
    
    if isinstance(value, str):
        try:
            # Try to convert to a number
            float(value)
            return True
        except ValueError:
            pass
    
    return False


def extract_from_calculation_steps(text, input_numbers):
    """
    Extract the final answer from calculation steps.
    
    This function looks for calculation steps that lead to a final answer.
    
    Args:
        text (str): The text to search for calculation steps
        input_numbers (list): The list of input numbers from the prompt
        
    Returns:
        The extracted answer or None
    """
    # Look for absolute difference steps with a clear final result
    # Pattern: "abs(X - Y) = Z" or "|X - Y| = Z" or "absolute difference = Z"
    abs_diff_patterns = [
        r'abs\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*=\s*([+-]?\d+(?:\.\d+)?)',
        r'\|\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*=\s*([+-]?\d+(?:\.\d+)?)',
        r'absolute\s+difference\s*(?:of|between)?\s*(?:[^=]*)\s*=\s*([+-]?\d+(?:\.\d+)?)'
    ]
    
    for pattern in abs_diff_patterns:
        steps = re.findall(pattern, text.lower())
        if steps:
            # Get the last step's result (likely the final answer)
            last_result = steps[-1]
            # If the result is a tuple (from the first two patterns), get the third element
            if isinstance(last_result, tuple) and len(last_result) >= 3:
                last_result = last_result[2]
            
            # Check if this is not just an input number
            if not is_input_number(last_result, input_numbers):
                return last_result
    
    # Look for subtraction steps that might indicate absolute difference
    # Pattern: "X - Y = Z" where Z is positive
    subtraction_steps = re.findall(r'([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)', text)
    
    if subtraction_steps:
        # Get the last step's result (likely the final answer)
        last_result = subtraction_steps[-1][2]
        
        # Check if this is not just an input number
        if not is_input_number(last_result, input_numbers):
            return last_result
    
    # Look for a sequence of calculations with intermediate results
    # Pattern: "First: |X - Y| = Z", "Then: Z + W = V", etc.
    step_pattern = r'(?:first|then|next|finally|step|lastly|after that)[:\s]*(?:[^=]*=\s*)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    step_matches = re.findall(step_pattern, text.lower())
    
    if step_matches and len(step_matches) > 0:
        # Get the last step's result
        last_result = step_matches[-1]
        
        # Check if this is not just an input number
        if not is_input_number(last_result, input_numbers):
            return last_result
    
    # Look for a clear final calculation step
    final_step = re.search(r'(?:final|result|answer|absolute difference)[:\s]*(?:[^=]*=\s*)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text.lower())
    if final_step:
        result = final_step.group(1)
        
        # Check if this is not just an input number
        if not is_input_number(result, input_numbers):
            return result
    
    return None

def extract_input_numbers_from_prompt(prompt):
    """
    Extract the input numbers from the prompt.
    
    Args:
        prompt (str): The prompt that was given to the LLM
        
    Returns:
        list: The list of input numbers
    """
    # Look for a list pattern in the prompt
    list_match = re.search(r'\[([-+]?\d+(?:,\s*[-+]?\d+)*)\]', prompt)
    if list_match:
        # Extract the numbers from the list
        numbers_str = list_match.group(1)
        # Split by comma and convert to numbers
        try:
            numbers = [int(num.strip()) for num in numbers_str.split(',')]
            return numbers
        except ValueError:
            pass
    
    return []


def is_input_number(number, input_numbers):
    """
    Check if a number is one of the input numbers or a simple combination of them.
    
    Args:
        number: The number to check
        input_numbers (list): The list of input numbers from the prompt
        
    Returns:
        bool: True if the number is one of the input numbers, False otherwise
    """
    if not input_numbers:
        return False
    
    # Convert to int or float for comparison
    if isinstance(number, str):
        try:
            number = int(float(number))
        except ValueError:
            try:
                number = float(number)
            except ValueError:
                return False
    
    # Check if the number is one of the input numbers
    if number in input_numbers:
        return True
    
    # Check if the number is a simple sum of input numbers
    if len(input_numbers) > 1 and number == sum(input_numbers):
        return True
    
    # Check if the number is one of the input numbers with a sign change
    if -number in input_numbers:
        return True
    
    return False


def clean_and_convert_to_number(text):
    """
    Clean the text and convert it to a number if possible.
    
    Args:
        text (str or int or float): The text or number to clean and convert
        
    Returns:
        int, float, or str: The converted number or the original text if conversion fails
    """
    if not text:
        return None
    
    # If text is already a number, return it
    if isinstance(text, (int, float)):
        return text
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None
    
    # Remove LaTeX formatting
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove markdown formatting (bold, italic, code)
    text = re.sub(r'[*_~`]', '', text)
    
    # Handle cases where there's a space between the minus sign and the number
    text = re.sub(r'-\s+(\d+)', r'-\1', text)
    
    # Remove commas from numbers (e.g., 1,234,567)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    # Handle LaTeX exponents (e.g., 10^{6})
    text = re.sub(r'(\d+)\^{(\d+)}', r'\1e\2', text)
    
    # Handle LaTeX times notation (e.g., 1.5 \times 10^{6})
    times_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*\\times\s*10\^{(\d+)}', text)
    if times_match:
        base = float(times_match.group(1))
        exponent = int(times_match.group(2))
        return base * (10 ** exponent)
    
    # Handle textual scientific notation (e.g., 1.5 x 10^6 or 1.5 × 10^6)
    sci_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*[x×]\s*10\^(\d+)', text)
    if sci_match:
        base = float(sci_match.group(1))
        exponent = int(sci_match.group(2))
        return base * (10 ** exponent)
    
    # Handle cases where the model provides a calculation like "X / Y = Z"
    calc_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*[÷/]\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)', text)
    if calc_match:
        return int(float(calc_match.group(3))) if '.' not in calc_match.group(3) else float(calc_match.group(3))
    
    # Handle cases where the model provides a calculation with multiple steps
    # Look for the last "=" in the text
    last_equal_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$', text)
    if last_equal_match:
        return int(float(last_equal_match.group(1))) if '.' not in last_equal_match.group(1) and 'e' not in last_equal_match.group(1).lower() else float(last_equal_match.group(1))
    
    # Try to convert to a number
    try:
        # Extract the first number from the text, including scientific notation
        number_match = re.search(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text)
        if number_match:
            number_str = number_match.group(1)
            # Convert to int or float as appropriate
            return int(float(number_str)) if '.' not in number_str and 'e' not in number_str.lower() else float(number_str)
    except ValueError:
        pass
    
    # Return the cleaned text if conversion fails
    return text


def extract_from_boxed_formats(text):
    """
    Extract answers from various boxed formats.
    
    This function looks for different LaTeX-style boxed formats that LLMs might use,
    including standard \boxed{}, parenthesized \(\boxed{}\), and bracketed \[\boxed{}\].
    
    Args:
        text (str): The text to search for boxed formats
        
    Returns:
        tuple: (extracted_answer, instruction_followed)
               extracted_answer is the content inside the box or None
               instruction_followed is True if the exact requested format was used
    """
    # Look for standard LaTeX boxed format: \boxed{answer}
    standard_boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
    
    # Look for standard LaTeX boxed format with space: \ boxed{answer}
    standard_boxed_with_space = re.findall(r'\\\s+boxed\{([^{}]*)\}', text)
    
    # Look for LaTeX boxed format with parentheses: \(\boxed{answer}\)
    paren_boxed = re.findall(r'\\[\(\[]\\boxed\{([^{}]*)\}\\[\)\]]', text)
    
    # Look for LaTeX boxed format with parentheses and space: \(\ boxed{answer}\)
    paren_boxed_with_space = re.findall(r'\\[\(\[]\\\s+boxed\{([^{}]*)\}\\[\)\]]', text)
    
    # Look for LaTeX boxed format with brackets: \[\boxed{answer}\]
    bracket_boxed = re.findall(r'\\[\[\]]\\boxed\{([^{}]*)\}\\[\[\]]', text)
    
    # Look for LaTeX boxed format with brackets and space: \[\ boxed{answer}\]
    bracket_boxed_with_space = re.findall(r'\\[\[\]]\\\s+boxed\{([^{}]*)\}\\[\[\]]', text)
    
    # Look for markdown-style boxed format: [boxed{answer}]
    markdown_boxed = re.findall(r'\[boxed\{([^{}]*)\}\]', text)
    
    # Look for markdown-style boxed format with space: [\ boxed{answer}]
    markdown_boxed_with_space = re.findall(r'\[\\\s+boxed\{([^{}]*)\}\]', text)
    
    # Look for alternative boxed formats that LLMs might use
    alt_boxed1 = re.findall(r'\\box\{([^{}]*)\}', text)
    alt_boxed2 = re.findall(r'\[\\boxed\{([^{}]*)\}\]', text)
    alt_boxed3 = re.findall(r'\(\boxed\{([^{}]*)\}\)', text)
    
    # Look for malformed boxed formats with extra backslashes
    malformed_boxed1 = re.findall(r'\\\s*boxed\{([^{}]*)\}', text)
    malformed_boxed2 = re.findall(r'\\\\boxed\{([^{}]*)\}', text)
    
    # Look for boxed formats with different brackets/braces
    alt_boxed4 = re.findall(r'\\boxed\[([^\[\]]*)\]', text)
    alt_boxed5 = re.findall(r'\\boxed\(([^()]*)\)', text)
    
    # Look for boxed formats with escaped characters
    escaped_boxed = re.findall(r'\\!?\\[\(\[]\\?(?:quotient|result|division\\)?=?\\boxed\{\{([^{}]*)\}\}\\[\)\]]', text)
    
    # Look for boxed formats with text formatting inside
    formatted_boxed = re.findall(r'\\boxed\{\\textbf\{([^{}]*)\}\}', text)
    formatted_boxed2 = re.findall(r'\\boxed\{\\text\{([^{}]*)\}\}', text)
    formatted_boxed3 = re.findall(r'\\boxed\{\\mathbf\{([^{}]*)\}\}', text)
    formatted_boxed4 = re.findall(r'\\boxed\{\\textcolor\{[^{}]*\}\{([^{}]*)\}\}', text)
    formatted_boxed5 = re.findall(r'\\boxed\{\\mathrm\{([^{}]*)\}\}', text)
    
    # Look for boxed formats with nested braces
    nested_boxed = re.findall(r'\\boxed\{\{([^{}]*)\}\}', text)
    
    # Look for text-based boxed formats
    text_boxed = re.findall(r'\\boxed\{answer\s*=\s*([^{}]*)\}', text)
    text_boxed2 = re.findall(r'\\boxed\{([^{}]*)\}', text)
    
    # Look for boxed formats with special characters
    special_boxed = re.findall(r'\\boxed\{[-+]?[^{}]*\}', text)
    if special_boxed:
        # Extract the content inside the boxed format
        special_boxed_content = []
        for match in special_boxed:
            inner_content = re.search(r'\\boxed\{([-+]?[^{}]*)\}', match)
            if inner_content:
                special_boxed_content.append(inner_content.group(1))
    else:
        special_boxed_content = []
    
    # Look for plain boxed format without LaTeX: boxed{answer}
    plain_boxed = re.findall(r'boxed\{([^{}]*)\}', text)
    
    # Look for alternative formats with "answer" keyword
    answer_boxed = re.findall(r'\\boxed\{\\?(?:answer|result|quotient)\s*(?:=|:)\s*([^{}]*)\}', text)
    
    # Look for formats with scientific notation
    sci_notation_boxed = re.findall(r'\\boxed\{([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)\}', text)
    
    # Look for formats with times symbol (×)
    times_boxed = re.findall(r'\\boxed\{([+-]?\d+(?:\.\d+)?)\\times10\^\{([+-]?\d+)\}\}', text)
    
    # Look for formats with fractions
    fraction_boxed = re.findall(r'\\boxed\{\\frac\{([^{}]*)\}\{([^{}]*)\}\}', text)
    
    # Look for formats with negative signs in different positions
    neg_boxed = re.findall(r'\\boxed\{-\s*([^{}]*)\}', text)
    neg_boxed2 = re.findall(r'\\boxed\{\\text\{-\}\s*([^{}]*)\}', text)
    neg_boxed3 = re.findall(r'\\boxed\{\\textbf\{-\}\s*([^{}]*)\}', text)
    
    # Look for formats with scientific notation in text form
    sci_text_boxed = re.findall(r'\\boxed\{([+-]?\d+(?:\.\d+)?)\s*[×x]\s*10\^{?(\d+)}?\}', text)
    
    # Look for formats with mixed notation (e.g., \boxed{-\textbf{3312}})
    mixed_boxed = re.findall(r'\\boxed\{-\\textbf\{([^{}]*)\}\}', text)
    mixed_boxed2 = re.findall(r'\\boxed\{-\\text\{([^{}]*)\}\}', text)
    mixed_boxed3 = re.findall(r'\\boxed\{-\\mathbf\{([^{}]*)\}\}', text)
    
    # Look for additional malformed boxed formats found in the false instruction cases
    text_boxed_format = re.findall(r'\\text\[boxed\]\s*([^\\]+)', text)
    boxed_format = re.findall(r'boxed\[([^\]]+)\]', text)
    boxed_answer_format = re.findall(r'boxed\{answer\}', text)
    boxed_double_brace = re.findall(r'\\boxed\{\{([^{}]+)\}\}', text)
    boxed_triple_brace = re.findall(r'\\boxed\{\{\{([^{}]+)\}\}\}', text)
    boxed_with_space = re.findall(r'\\boxed\s+\{([^{}]+)\}', text)
    boxed_with_underscore = re.findall(r'\\boxed\_\{([^{}]+)\}', text)
    boxed_with_dollar = re.findall(r'\$\\boxed\{([^{}]+)\}\$', text)
    
    # Combine all matches
    all_matches = (standard_boxed + standard_boxed_with_space + 
                  paren_boxed + paren_boxed_with_space + 
                  bracket_boxed + bracket_boxed_with_space + 
                  markdown_boxed + markdown_boxed_with_space + 
                  alt_boxed1 + alt_boxed2 + alt_boxed3 + alt_boxed4 + alt_boxed5 + 
                  malformed_boxed1 + malformed_boxed2 + escaped_boxed +
                  formatted_boxed + formatted_boxed2 + formatted_boxed3 + formatted_boxed4 + formatted_boxed5 +
                  nested_boxed + text_boxed + special_boxed_content +
                  plain_boxed + answer_boxed + sci_notation_boxed +
                  text_boxed_format + boxed_format + boxed_double_brace + boxed_triple_brace +
                  boxed_with_space + boxed_with_underscore + boxed_with_dollar)
    
    # Process negative boxed formats
    for match in neg_boxed + neg_boxed2 + neg_boxed3:
        try:
            value = "-" + match
            all_matches.append(value)
        except (ValueError, TypeError):
            pass
    
    # Process mixed boxed formats
    for match in mixed_boxed + mixed_boxed2 + mixed_boxed3:
        try:
            value = "-" + match
            all_matches.append(value)
        except (ValueError, TypeError):
            pass
    
    # Process times notation matches
    for match in times_boxed:
        if len(match) == 2:
            base, exponent = match
            try:
                value = float(base) * (10 ** int(exponent))
                all_matches.append(str(value))
            except (ValueError, TypeError):
                pass
    
    # Process scientific text notation matches
    for match in sci_text_boxed:
        if len(match) == 2:
            base, exponent = match
            try:
                value = float(base) * (10 ** int(exponent))
                all_matches.append(str(value))
            except (ValueError, TypeError):
                pass
    
    # Process fraction matches
    for match in fraction_boxed:
        if len(match) == 2:
            numerator, denominator = match
            try:
                value = float(numerator) / float(denominator)
                all_matches.append(str(value))
            except (ValueError, TypeError, ZeroDivisionError):
                pass
    
    # Special case for boxed_answer_format - try to find a number near it
    if boxed_answer_format:
        # Look for a number near the boxed{answer} format
        nearby_number = re.search(r'boxed\{answer\}[^0-9]*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text)
        if nearby_number:
            all_matches.append(nearby_number.group(1))
    
    if all_matches:
        # Return the last match as it's likely the final answer
        return all_matches[-1].strip(), True
    
    return None, False


def extract_from_explicit_statements(text):
    """
    Extract answers from explicit statements like "Answer: X" or "The result is X".
    
    This function looks for common patterns that LLMs use to indicate their final answer.
    
    Args:
        text (str): The text to search for explicit answer statements
        
    Returns:
        tuple: (extracted_answer, instruction_followed)
               extracted_answer is the content after the statement or None
               instruction_followed is False since the requested format wasn't used
    """
    # Common patterns for explicit answer statements (generic, not specific to any task)
    patterns = [
        # Answer: X or The answer is X
        r'(?:^|\n|\s)(?:answer|the answer is|final answer|the final answer is)[:\s]+([^\n\.]+)',
        # The answer is \\(X\\)
        r'(?:^|\n|\s)(?:the answer is)[:\s]+\\\\[\(\[]([^\\]+)\\\\[\)\]]',
        # Therefore, X is the answer
        r'(?:^|\n|\s)(?:therefore,?)[:\s]+([^\n\.]+)(?:\s+is the answer)',
        # Therefore, the answer is X
        r'(?:^|\n|\s)(?:therefore,? the answer is)[:\s]+([^\n\.]+)',
        # Thus, the answer is X
        r'(?:^|\n|\s)(?:thus,? the answer is)[:\s]+([^\n\.]+)',
        # So, the answer is X
        r'(?:^|\n|\s)(?:so,? the answer is)[:\s]+([^\n\.]+)',
        # Hence, the answer is X
        r'(?:^|\n|\s)(?:hence,? the answer is)[:\s]+([^\n\.]+)',
        # The result is X
        r'(?:^|\n|\s)(?:the result is)[:\s]+([^\n\.]+)',
        # The value is X
        r'(?:^|\n|\s)(?:the value is)[:\s]+([^\n\.]+)',
        # The quotient is X
        r'(?:^|\n|\s)(?:the quotient is)[:\s]+([^\n\.]+)',
        # The number is X
        r'(?:^|\n|\s)(?:the number is)[:\s]+([^\n\.]+)',
        # The output is X
        r'(?:^|\n|\s)(?:the output is)[:\s]+([^\n\.]+)',
        # The solution is X
        r'(?:^|\n|\s)(?:the solution is)[:\s]+([^\n\.]+)',
        # X is the answer
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the answer)',
        # X is the result
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the result)',
        # X is the value
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the value)',
        # X is the number
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the number)',
        # X is the output
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the output)',
        # X is the solution
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the solution)',
        # X is the product
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the product)',
        # The absolute difference of the numbers is X
        r'(?:^|\n|\s)(?:the absolute difference (?:of|between) the (?:given )?(?:numbers|list) is)[:\s]+([^\n\.]+)',
        # The result of finding the absolute difference is X
        r'(?:^|\n|\s)(?:the result of finding the absolute difference is)[:\s]+([^\n\.]+)',
        # The result of the absolute difference is X
        r'(?:^|\n|\s)(?:the result of the absolute difference is)[:\s]+([^\n\.]+)',
        # The absolute difference result is X
        r'(?:^|\n|\s)(?:the absolute difference result is)[:\s]+([^\n\.]+)',
        # Finding the absolute difference gives X
        r'(?:^|\n|\s)(?:finding the absolute difference gives)[:\s]+([^\n\.]+)',
        # The absolute difference between number 1 and number 2 is X
        r'(?:^|\n|\s)(?:the absolute difference between number 1 and number 2 is)[:\s]+([^\n\.]+)',
        # When we find the absolute difference, we get X
        r'(?:^|\n|\s)(?:when we find the absolute difference,? we get)[:\s]+([^\n\.]+)',
        # When we calculate the absolute difference between number 1 and number 2, we get X
        r'(?:^|\n|\s)(?:when we calculate the absolute difference between number 1 and number 2,? we get)[:\s]+([^\n\.]+)',
        # The absolute difference between the two numbers is X
        r'(?:^|\n|\s)(?:the absolute difference between the two numbers is)[:\s]+([^\n\.]+)',
        # Patterns for backticked numbers (absolute difference-specific)
        r'(?:^|\n|\s)(?:the absolute difference (?:of|between) the (?:given )?(?:numbers|list) is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the result of finding the absolute difference is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the result of the absolute difference is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the absolute difference result is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:finding the absolute difference gives)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the absolute difference between number 1 and number 2 is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:when we find the absolute difference,? we get)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:when we calculate the absolute difference between number 1 and number 2,? we get)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the absolute difference between the two numbers is)[:\s]+`([^`]+)`',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # Extract the actual match content (might be a tuple from regex groups)
            match = matches[-1]
            if isinstance(match, tuple):
                match = match[0]
            
            # Clean up the match
            cleaned_match = match.strip()
            
            # Return the last match as it's likely the final answer
            return cleaned_match, False
    
    return None, False


def extract_from_latex_math(text):
    """
    Extract answers from LaTeX math expressions.
    
    This function looks for LaTeX math expressions that might contain the answer.
    
    Args:
        text (str): The text to search for LaTeX math expressions
        
    Returns:
        The extracted answer or None
    """
    # Look for LaTeX math expressions: $X$, $$X$$, \(X\), \[X\]
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
            # Return the last match as it's likely the final answer
            return matches[-1].strip()
    
    return None


def extract_from_code_blocks(text):
    """
    Extract answers from code blocks.
    
    This function looks for code blocks that might contain the answer.
    
    Args:
        text (str): The text to search for code blocks
        
    Returns:
        The extracted answer or None
    """
    # Look for code blocks: ```X```, `X`
    patterns = [
        r'```(?:python|plaintext)?\s*\n([^`]+)\n```',
        r'`([^`]+)`',
    ]
    
    # Look for output blocks that might contain the answer
    output_patterns = [
        r'```\s*\n([^`]+)\n```',  # Output in code block
        r'Output:\s*\n([^\n]+)',  # Output with label
        r'output:\s*\n([^\n]+)',  # Output with lowercase label
        r'The output is:\s*\n([^\n]+)',  # Output with full sentence
        r'Running this code will output:\s*\n([^\n]+)',  # Output with running code
        r'When you run this code it will output:\s*\n([^\n]+)',  # Output with running code
        r'The result is:\s*\n([^\n]+)',  # Result with label
        r'Result:\s*\n([^\n]+)',  # Result with label
    ]
    
    # First, try to extract from output blocks
    for pattern in output_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                # Look for boxed format in the output (with or without space after backslash)
                boxed_match = re.search(r'\\(?:\s*)boxed\{([^{}]+)\}', match)
                if boxed_match:
                    return boxed_match.group(1).strip()
                
                # Look for plain numbers in the output
                number_match = re.search(r'[+-]?\d+(?:\.\d+)?', match)
                if number_match:
                    return number_match.group(0).strip()
    
    # Then, try to extract from code blocks
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Look for answers in the code block
            for match in matches:
                # Look for print statements with boxed format (allowing space after backslash and triple braces)
                boxed_print_match = re.search(r'print\s*\(\s*f[\"\']\\(?:\\)?(?:\s*)boxed\{(?:\{+)([^{}]+)(?:\}+)\}[\"\']', match)
                if boxed_print_match:
                    # Check if the extracted value is a number
                    try:
                        value = boxed_print_match.group(1).strip()
                        if re.match(r'^[+-]?\d+(?:\.\d+)?$', value):
                            return value
                    except:
                        pass
                
                # Look for print statements with the word "absolute difference" or "abs diff"
                abs_diff_print_match = re.search(r'print\s*\(\s*["\']?(?:.*?(?:absolute difference|abs diff|abs|difference)[^"\']*?)?([+-]?\d+(?:\.\d+)?)["\']?\s*\)', match)
                if abs_diff_print_match:
                    return abs_diff_print_match.group(1).strip()
                
                # Look for print statements with comma-separated arguments
                comma_print_match = re.search(r'print\s*\(\s*["\'].*?["\'],\s*([+-]?\d+(?:\.\d+)?)\s*\)', match)
                if comma_print_match:
                    return comma_print_match.group(1).strip()
                
                # Look for regular print statements
                print_match = re.search(r'print\s*\(\s*["\']?([^"\']+)["\']?\s*\)', match)
                if print_match:
                    # Check if the extracted value is a number
                    try:
                        value = print_match.group(1).strip()
                        if re.match(r'^[+-]?\d+(?:\.\d+)?$', value):
                            return value
                    except:
                        pass
                
                # Look for f-string print statements
                f_print_match = re.search(r'print\s*\(\s*f[\"\'].*?([+-]?\d+(?:\.\d+)?).*?[\"\']\s*\)', match)
                if f_print_match:
                    return f_print_match.group(1).strip()
                
                # Look for return statements
                return_match = re.search(r'return\s+([+-]?\d+(?:\.\d+)?)', match)
                if return_match:
                    return return_match.group(1).strip()
                
                # Look for any variable assignments with a number
                var_assign_match = re.search(r'\b\w+\s*=\s*([+-]?\d+(?:\.\d+)?)(?!\s*\[)', match)
                if var_assign_match:
                    return var_assign_match.group(1).strip()
                
                # Look for absolute difference variable assignments with final values
                abs_diff_assign_match = re.search(r'(?:abs_diff|absolute_difference|difference|result|total)\s*=\s*([+-]?\d+(?:\.\d+)?)(?!\s*\[)', match)
                if abs_diff_assign_match:
                    return abs_diff_assign_match.group(1).strip()
                
                # Look for abs() function calls
                abs_func_match = re.search(r'abs\s*\(\s*(?:[^-]+\s*-\s*[^)]+|[^)]+)\s*\)\s*(?:==?|is)\s*([+-]?\d+(?:\.\d+)?)', match)
                if abs_func_match:
                    return abs_func_match.group(1).strip()
                
                # Look for variable assignments with "answer" or "result"
                assign_match = re.search(r'(?:answer|result)\s*=\s*([+-]?\d+(?:\.\d+)?)', match)
                if assign_match:
                    return assign_match.group(1).strip()
    
    # Look for the absolute difference in the text
    abs_diff_match = re.search(r'(?:absolute difference|abs diff|difference)(?:\s+(?:of|between))?\s+(?:is|=)\s+([+-]?\d+(?:\.\d+)?)', text.lower())
    if abs_diff_match:
        return abs_diff_match.group(1).strip()
    
    # Look for statements about the absolute difference
    abs_diff_statement_match = re.search(r'(?:the )?(?:absolute difference|abs diff|difference)(?:\s+(?:of|between))?\s+(?:the )?(?:list|array|numbers|two numbers)\s+is\s+([+-]?\d+(?:\.\d+)?)', text.lower())
    if abs_diff_statement_match:
        return abs_diff_statement_match.group(1).strip()
    
    return None


def extract_from_last_line(text):
    """
    Extract answers from the last line or sentence.
    
    This function looks for potential answers in the last line or sentence of the text.
    
    Args:
        text (str): The text to search for answers in the last line or sentence
        
    Returns:
        The extracted answer or None
    """
    # Handle literal \n characters in the text
    text = text.replace('\\n', '\n')
    
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Check the last few lines for potential answers
    for i in range(min(3, len(lines))):
        line = lines[-(i+1)].strip()
        if line:
            # Look for patterns like "Absolute difference of the numbers is: X"
            abs_diff_match = re.search(r'(?:absolute difference|abs diff|difference)(?:\s+(?:of|between))?\s+(?:the )?(?:list|array|numbers|two numbers)?\s+is:?\s+([+-]?\d+(?:\.\d+)?)', line.lower())
            if abs_diff_match:
                return abs_diff_match.group(1).strip()
            
            # Look for patterns like "X" or "X." at the end of the line
            answer_match = re.search(r'[\"\']([^\"\'.]+)[\"\']\.?$|([+-]?\d+(?:\.\d+)?)\.?$', line)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()
    
    # Split the text into sentences
    sentences = re.split(r'[.!?]', text)
    
    # Check the last few sentences for potential answers
    for i in range(min(3, len(sentences))):
        sentence = sentences[-(i+1)].strip()
        if sentence:
            # Look for patterns like "Absolute difference of the numbers is: X"
            abs_diff_match = re.search(r'(?:absolute difference|abs diff|difference)(?:\s+(?:of|between))?\s+(?:the )?(?:list|array|numbers|two numbers)?\s+is:?\s+([+-]?\d+(?:\.\d+)?)', sentence.lower())
            if abs_diff_match:
                return abs_diff_match.group(1).strip()
            
            # Look for patterns like "X" or "X" at the end of the sentence
            answer_match = re.search(r'[\"\']([^\"\'.]+)[\"\']$|([+-]?\d+(?:\.\d+)?)$', sentence)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()
    
    return None


def extract_from_scientific_notation(text):
    """
    Extract answers from scientific notation formats.
    
    This function looks for scientific notation formats that might contain the answer.
    
    Args:
        text (str): The text to search for scientific notation
        
    Returns:
        The extracted answer or None
    """
    # Look for scientific notation patterns
    patterns = [
        # Standard scientific notation: 1.23e+4 or 1.23E-4
        r'([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)',
        # Scientific notation with × symbol: 1.23 × 10^4
        r'([+-]?\d+(?:\.\d+)?)\s*[×x]\s*10\^{?(\d+)}?',
        # Scientific notation with times word: 1.23 times 10^4
        r'([+-]?\d+(?:\.\d+)?)\s*times\s*10\^{?(\d+)}?',
        # Scientific notation with LaTeX: 1.23 \times 10^{4}
        r'([+-]?\d+(?:\.\d+)?)\s*\\times\s*10\^{(\d+)}',
        # Scientific notation with dot notation: 1.23 \cdot 10^{4}
        r'([+-]?\d+(?:\.\d+)?)\s*\\cdot\s*10\^{(\d+)}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Process the matches
            for match in matches:
                if isinstance(match, tuple):
                    # For patterns with base and exponent as separate groups
                    base, exponent = match
                    try:
                        value = float(base) * (10 ** int(exponent))
                        return str(value)
                    except (ValueError, TypeError):
                        pass
                else:
                    # For patterns with the full notation as a single group
                    try:
                        value = float(match)
                        return str(value)
                    except (ValueError, TypeError):
                        pass
    
    # Look for scientific notation in text descriptions
    text_patterns = [
        # "X times 10 to the power of Y"
        r'([+-]?\d+(?:\.\d+)?)\s*times\s*10\s*to\s*the\s*(?:power\s*of)?\s*(\d+)',
        # "X multiplied by 10 to the power of Y"
        r'([+-]?\d+(?:\.\d+)?)\s*multiplied\s*by\s*10\s*to\s*the\s*(?:power\s*of)?\s*(\d+)',
    ]
    
    for pattern in text_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Process the matches
            for base, exponent in matches:
                try:
                    value = float(base) * (10 ** int(exponent))
                    return str(value)
                except (ValueError, TypeError):
                    pass
    
    return None


def extract_from_array_format(text):
    """
    Extract answers from array or list formats.
    
    This function looks for array or list formats that might contain the answer.
    
    Args:
        text (str): The text to search for array or list formats
        
    Returns:
        The extracted answer or None
    """
    # Look for array or list patterns
    patterns = [
        # Array with single element: [X]
        r'\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]',
        # Array with multiple elements: [X, Y, Z]
        r'\[\s*(?:[^,\[\]]*,\s*)*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:,\s*[^,\[\]]*)*\s*\]',
        # Nested array: [[X]]
        r'\[\s*\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]\s*\]',
        # Array with label: [absolute difference: X]
        r'\[\s*(?:absolute difference|abs diff|difference|result|answer)(?:\s*(?:is|:|=)\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]',
        # Parentheses with single element: (X)
        r'\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)',
        # Parentheses with multiple elements: (X, Y, Z)
        r'\(\s*(?:[^,()]*,\s*)*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:,\s*[^,()]*)*\s*\)',
        # Nested parentheses: ((X))
        r'\(\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)\s*\)',
        # Parentheses with label: (absolute difference: X)
        r'\(\s*(?:absolute difference|abs diff|difference|result|answer)(?:\s*(?:is|:|=)\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)',
        # Curly braces with single element: {X}
        r'\{\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}',
        # Curly braces with multiple elements: {X, Y, Z}
        r'\{\s*(?:[^,{}]*,\s*)*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:,\s*[^,{}]*)*\s*\}',
        # Nested curly braces: {{X}}
        r'\{\s*\{\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}\s*\}',
        # Curly braces with label: {absolute difference: X}
        r'\{\s*(?:absolute difference|abs diff|difference|result|answer)(?:\s*(?:is|:|=)\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last match as it's likely the final answer
            return matches[-1].strip()
    
    return None


def extract_from_markdown_formatting(text):
    """
    Extract answers from markdown formatting like bold or italic.
    
    This function looks for markdown-formatted text that might contain the answer.
    
    Args:
        text (str): The text to search for markdown formatting
        
    Returns:
        The extracted answer or None
    """
    # Look for bold formatting with asterisks: **X**
    bold_asterisks = re.findall(r'\*\*([^*]+)\*\*', text)
    
    # Look for bold formatting with underscores: __X__
    bold_underscores = re.findall(r'__([^_]+)__', text)
    
    # Look for italic formatting with asterisks: *X*
    italic_asterisks = re.findall(r'(?<!\*)\*([^*]+)\*(?!\*)', text)
    
    # Look for italic formatting with underscores: _X_
    italic_underscores = re.findall(r'(?<!_)_([^_]+)_(?!_)', text)
    
    # Combine all matches
    all_matches = bold_asterisks + bold_underscores + italic_asterisks + italic_underscores
    
    if all_matches:
        # Return the last match as it's likely the final answer
        return all_matches[-1].strip()
    
    return None


def extract_plain_number(text):
    """
    Extract plain numbers from very short responses.
    
    This function is used as a last resort for very short responses that might just be the number.
    
    Args:
        text (str): The text to search for plain numbers
        
    Returns:
        The extracted number or None
    """
    # If the response is very short (less than 10 characters)
    if len(text.strip()) < 10:
        # Try to extract a number
        number_match = re.search(r'^[+-]?\d+(?:\.\d+)?$', text.strip())
        if number_match:
            number_str = number_match.group(0)
            # Convert to int or float as appropriate
            return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number at the end of the text
    number_match = re.search(r'([+-]?\d+(?:\.\d+)?)$', text.strip())
    if number_match:
        number_str = number_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number in square brackets
    bracket_match = re.search(r'\[([+-]?\d+(?:\.\d+)?)\]', text)
    if bracket_match:
        number_str = bracket_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number in parentheses
    paren_match = re.search(r'\(([+-]?\d+(?:\.\d+)?)\)', text)
    if paren_match:
        number_str = paren_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number preceded by "is" or "="
    is_match = re.search(r'(?:is|=)\s*([+-]?\d+(?:\.\d+)?)', text)
    if is_match:
        number_str = is_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number preceded by "absolute difference" or related terms
    abs_diff_match = re.search(r'(?:absolute difference|abs diff|difference|abs)(?:\s+(?:of|between|the|numbers|value|result))?\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?)', text.lower())
    if abs_diff_match:
        number_str = abs_diff_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    return None
