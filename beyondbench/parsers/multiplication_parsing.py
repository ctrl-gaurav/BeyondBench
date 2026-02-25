import re

def parse_multiplication_answer(clean_response, prompt=None):
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
    # Look for multiplication steps with a clear final result
    # Pattern: "X * Y = Z" followed by "Z * W = V" etc.
    steps = re.findall(r'([+-]?\d+(?:\.\d+)?)\s*[×*]\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)', text)
    
    if steps:
        # Get the last step's result (likely the final answer)
        last_result = steps[-1][2]
        
        # Check if this is not just an input number
        if not is_input_number(last_result, input_numbers):
            return last_result
    
    # Look for a sequence of calculations with intermediate results
    # Pattern: "First: X * Y = Z", "Then: Z * W = V", etc.
    step_pattern = r'(?:first|then|next|finally|step|lastly|after that)[:\s]*(?:[^=]*=\s*)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    step_matches = re.findall(step_pattern, text.lower())
    
    if step_matches and len(step_matches) > 0:
        # Get the last step's result
        last_result = step_matches[-1]
        
        # Check if this is not just an input number
        if not is_input_number(last_result, input_numbers):
            return last_result
    
    # Look for a clear final calculation step
    final_step = re.search(r'(?:final|result|answer|product)[:\s]*(?:[^=]*=\s*)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text.lower())
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
    
    # Handle cases where the model provides a calculation like "X * Y = Z"
    calc_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*[×*]\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)', text)
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
    escaped_boxed = re.findall(r'\\!?\\[\(\[]\\?(?:product|result|multiplication\\)?=?\\boxed\{\{([^{}]*)\}\}\\[\)\]]', text)
    
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
        for match in special_boxed:
            inner_content = re.search(r'\\boxed\{([-+]?[^{}]*)\}', match)
            if inner_content:
                special_boxed_content = [inner_content.group(1)]
    else:
        special_boxed_content = []
    
    # Look for plain boxed format without LaTeX: boxed{answer}
    plain_boxed = re.findall(r'boxed\{([^{}]*)\}', text)
    
    # Look for alternative formats with "answer" keyword
    answer_boxed = re.findall(r'\\boxed\{\\?(?:answer|result|product)\s*(?:=|:)\s*([^{}]*)\}', text)
    
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
        # The product is X
        r'(?:^|\n|\s)(?:the product is)[:\s]+([^\n\.]+)',
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
        # The product of the numbers is X
        r'(?:^|\n|\s)(?:the product of the (?:given )?(?:numbers|list) is)[:\s]+([^\n\.]+)',
        # The product of all numbers is X
        r'(?:^|\n|\s)(?:the product of all (?:the )?(?:given )?numbers is)[:\s]+([^\n\.]+)',
        # The product of these numbers is X
        r'(?:^|\n|\s)(?:the product of these numbers is)[:\s]+([^\n\.]+)',
        # The product of the list is X
        r'(?:^|\n|\s)(?:the product of the (?:given )?list is)[:\s]+([^\n\.]+)',
        # Multiplying these numbers gives X
        r'(?:^|\n|\s)(?:multiplying these numbers gives)[:\s]+([^\n\.]+)',
        # Multiplying all numbers gives X
        r'(?:^|\n|\s)(?:multiplying all (?:the )?numbers gives)[:\s]+([^\n\.]+)',
        # The result of multiplying the numbers is X
        r'(?:^|\n|\s)(?:the result of multiplying the numbers is)[:\s]+([^\n\.]+)',
        # The result of the multiplication is X
        r'(?:^|\n|\s)(?:the result of the multiplication is)[:\s]+([^\n\.]+)',
        # When we multiply all numbers, we get X
        r'(?:^|\n|\s)(?:when we multiply all (?:the )?numbers,? we get)[:\s]+([^\n\.]+)',
        # The multiplication result is X
        r'(?:^|\n|\s)(?:the multiplication result is)[:\s]+([^\n\.]+)',
        # Patterns for backticked numbers
        r'(?:^|\n|\s)(?:the product of the (?:given )?(?:numbers|list) is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the product of all (?:the )?(?:given )?numbers is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the product of these numbers is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the product of the (?:given )?list is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:multiplying these numbers gives)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:multiplying all (?:the )?numbers gives)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the result of multiplying the numbers is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the result of the multiplication is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:when we multiply all (?:the )?numbers,? we get)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the multiplication result is)[:\s]+`([^`]+)`',
        # Patterns for final statements
        r'(?:^|\n|\s)(?:so,? the final product is)[:\s]+([^\n\.]+)',
        r'(?:^|\n|\s)(?:thus,? the final product is)[:\s]+([^\n\.]+)',
        r'(?:^|\n|\s)(?:therefore,? the final product is)[:\s]+([^\n\.]+)',
        r'(?:^|\n|\s)(?:hence,? the final product is)[:\s]+([^\n\.]+)',
        r'(?:^|\n|\s)(?:the final product is)[:\s]+([^\n\.]+)',
        # Patterns for scientific notation
        r'(?:^|\n|\s)(?:the product is)[:\s]+([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)',
        r'(?:^|\n|\s)(?:the product of the numbers is)[:\s]+([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)',
        r'(?:^|\n|\s)(?:the result is)[:\s]+([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)',
        # Patterns for formatted answers
        r'(?:^|\n|\s)(?:the product is)[:\s]+\\boxed\{([^{}]+)\}',
        r'(?:^|\n|\s)(?:the product of the numbers is)[:\s]+\\boxed\{([^{}]+)\}',
        r'(?:^|\n|\s)(?:the result is)[:\s]+\\boxed\{([^{}]+)\}',
        # Patterns for answers with commas
        r'(?:^|\n|\s)(?:the product is)[:\s]+([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?)',
        r'(?:^|\n|\s)(?:the product of the numbers is)[:\s]+([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?)',
        r'(?:^|\n|\s)(?:the result is)[:\s]+([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?)',
        # Patterns for direct number statements
        r'(?:^|\n|\s)(?:product|answer|result)(?:\s*=\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
        r'(?:^|\n|\s)(?:product|answer|result)(?:\s*:\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
        # Patterns for standalone numbers
        r'(?:^|\n|\s)(?:^|\n)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:$|\n)',
        # Patterns for numbers in quotes
        r'(?:^|\n|\s)(?:"|\'|")([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)("|\'|")',
        # Patterns for numbers in brackets
        r'(?:^|\n|\s)\[([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\]',
        # Patterns for numbers in parentheses
        r'(?:^|\n|\s)\(([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)',
        # Patterns for numbers with "answer" label
        r'(?:^|\n|\s)(?:answer|result|product)(?:\s*:\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
        # Patterns for numbers with "answer" label in various formats
        r'(?:^|\n|\s)(?:answer|result|product)(?:\s*=\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
        r'(?:^|\n|\s)(?:answer|result|product)(?:\s*is\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
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
    
    # Look for product-specific LaTeX expressions
    product_patterns = [
        r'\$([^$]*product[^$]*=\s*[+-]?\d+[^$]*)\$',
        r'\$\$([^$]*product[^$]*=\s*[+-]?\d+[^$]*)\$\$',
        r'\\[\(\[]([^\\]*product[^\\]*=\s*[+-]?\d+[^\\]*)\\[\)\]]',
        r'\\begin\{equation\}([^\\]*product[^\\]*=\s*[+-]?\d+[^\\]*)\\end\{equation\}',
        r'\\begin\{align\}([^\\]*product[^\\]*=\s*[+-]?\d+[^\\]*)\\end\{align\}',
        r'\\begin\{math\}([^\\]*product[^\\]*=\s*[+-]?\d+[^\\]*)\\end\{math\}',
    ]
    
    # First try product-specific patterns
    for pattern in product_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Extract the number after "product = " or similar
            for match in matches:
                number_match = re.search(r'product[^=]*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', match, re.IGNORECASE)
                if number_match:
                    return number_match.group(1).strip()
    
    # Then try general patterns
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Look for product = X in the match
            for match in matches:
                number_match = re.search(r'product[^=]*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', match, re.IGNORECASE)
                if number_match:
                    return number_match.group(1).strip()
            
            # If no product = X found, look for the last number in the expression that's clearly labeled
            for match in matches:
                # Look for patterns like "= X" at the end of the expression
                number_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$', match.strip())
                if number_match:
                    return number_match.group(1).strip()
    
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
                
                # Look for product-specific statements in the output
                product_match = re.search(r'(?:product|result|answer)(?:\s+(?:is|=))\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', match, re.IGNORECASE)
                if product_match:
                    return product_match.group(1).strip()
    
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
                        if re.match(r'^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$', value):
                            return value
                    except:
                        pass
                
                # Look for print statements with the word "product" or "multiplication"
                product_print_match = re.search(r'print\s*\(\s*["\']?(?:.*?(?:product|multiplication|multiply|result)[^"\']*?)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)["\']?\s*\)', match, re.IGNORECASE)
                if product_print_match:
                    return product_print_match.group(1).strip()
                
                # Look for print statements with "The product is" or similar
                product_statement_match = re.search(r'print\s*\(\s*["\'](?:The|the)?\s*(?:product|result|answer)(?:\s+(?:is|=))\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)["\']?\s*\)', match, re.IGNORECASE)
                if product_statement_match:
                    return product_statement_match.group(1).strip()
                
                # Look for return statements with product
                return_product_match = re.search(r'return\s+(?:.*?(?:product|result|answer)[^"\']*?)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', match, re.IGNORECASE)
                if return_product_match:
                    return return_product_match.group(1).strip()
                
                # Look for product variable assignments with final values
                product_assign_match = re.search(r'(?:product|result|answer|multiplication|total)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?!\s*\[)', match, re.IGNORECASE)
                if product_assign_match:
                    return product_assign_match.group(1).strip()
                
                # Look for numbers in array/list context that might be the product
                array_match = re.search(r'\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]', match)
                if array_match:
                    return array_match.group(1).strip()
    
    # Look for the product in the text with clear labeling
    product_match = re.search(r'(?:product|multiplication|result)(?:\s+of)?\s+(?:is|=)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text.lower())
    if product_match:
        return product_match.group(1).strip()
    
    # Look for statements about the product with clear labeling
    product_statement_match = re.search(r'(?:the )?(?:product|multiplication|result)(?:\s+of)?\s+(?:the )?(?:list|array|numbers)\s+is\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text.lower())
    if product_statement_match:
        return product_statement_match.group(1).strip()
    
    # Look for step-by-step calculations with a final result
    calculation_steps = re.findall(r'([+-]?\d+(?:\.\d+)?)\s*[×*]\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)', text)
    if calculation_steps:
        # Get the last calculation step (likely the final result)
        last_step = calculation_steps[-1]
        if len(last_step) >= 3:
            return last_step[2].strip()
    
    # Look for a sequence of calculations that might represent steps
    calculation_sequence = re.findall(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[×*]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text)
    if calculation_sequence:
        # Get the last calculation (likely the final result)
        last_calculation = calculation_sequence[-1]
        if len(last_calculation) >= 3:
            return last_calculation[2].strip()
    
    # Look for a number in a list context that might be the product
    list_product_match = re.search(r'\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]', text)
    if list_product_match:
        return list_product_match.group(1).strip()
    
    # Look for a number after a multiplication symbol (×) that might be the product
    mult_symbol_match = re.search(r'[×*]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$', text)
    if mult_symbol_match:
        return mult_symbol_match.group(1).strip()
    
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
            # Look for patterns like "Product of the numbers is: X"
            product_match = re.search(r'(?:product|multiplication|result)(?:\s+of)?\s+(?:the )?(?:list|array|numbers)?\s+is:?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line.lower())
            if product_match:
                return product_match.group(1).strip()
            
            # Look for patterns like "Therefore, the product is X"
            therefore_match = re.search(r'(?:therefore|thus|hence|so),?\s+(?:the )?(?:product|result|answer)\s+is:?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line.lower())
            if therefore_match:
                return therefore_match.group(1).strip()
            
            # Look for patterns like "The final product is X"
            final_match = re.search(r'(?:the )?final\s+(?:product|result|answer)\s+is:?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line.lower())
            if final_match:
                return final_match.group(1).strip()
    
    # Split the text into sentences
    sentences = re.split(r'[.!?]', text)
    
    # Check the last few sentences for potential answers
    for i in range(min(3, len(sentences))):
        sentence = sentences[-(i+1)].strip()
        if sentence:
            # Look for patterns like "Product of the numbers is: X"
            product_match = re.search(r'(?:product|multiplication|result)(?:\s+of)?\s+(?:the )?(?:list|array|numbers)?\s+is:?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', sentence.lower())
            if product_match:
                return product_match.group(1).strip()
            
            # Look for patterns like "Therefore, the product is X"
            therefore_match = re.search(r'(?:therefore|thus|hence|so),?\s+(?:the )?(?:product|result|answer)\s+is:?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', sentence.lower())
            if therefore_match:
                return therefore_match.group(1).strip()
            
            # Look for patterns like "The final product is X"
            final_match = re.search(r'(?:the )?final\s+(?:product|result|answer)\s+is:?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', sentence.lower())
            if final_match:
                return final_match.group(1).strip()
    
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
    
    # Look for code formatting with backticks: `X`
    code_backticks = re.findall(r'`([^`]+)`', text)
    
    # Combine all matches
    all_matches = bold_asterisks + bold_underscores + italic_asterisks + italic_underscores + code_backticks
    
    if all_matches:
        # Look for matches that contain product-related terms
        product_matches = []
        for match in all_matches:
            if re.search(r'(?:product|result|answer|multiplication)', match.lower()):
                product_matches.append(match)
        
        if product_matches:
            # Extract the number from the product-related match
            for match in product_matches:
                number_match = re.search(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', match)
                if number_match:
                    return number_match.group(1).strip()
        
        # If no product-related matches, check if any match is just a number
        for match in all_matches:
            if re.match(r'^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$', match.strip()):
                return match.strip()
    
    # Look for numbers in brackets or parentheses that might be the answer
    bracketed_number = re.search(r'\[([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\]', text)
    if bracketed_number:
        return bracketed_number.group(1).strip()
    
    parenthesized_number = re.search(r'\(([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)', text)
    if parenthesized_number:
        return parenthesized_number.group(1).strip()
    
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
        # Array with label: [product: X]
        r'\[\s*(?:product|result|answer|multiplication)(?:\s*(?:is|:|=)\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]',
        # Parentheses with single element: (X)
        r'\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)',
        # Parentheses with multiple elements: (X, Y, Z)
        r'\(\s*(?:[^,()]*,\s*)*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:,\s*[^,()]*)*\s*\)',
        # Nested parentheses: ((X))
        r'\(\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)\s*\)',
        # Parentheses with label: (product: X)
        r'\(\s*(?:product|result|answer|multiplication)(?:\s*(?:is|:|=)\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)',
        # Curly braces with single element: {X}
        r'\{\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}',
        # Curly braces with multiple elements: {X, Y, Z}
        r'\{\s*(?:[^,{}]*,\s*)*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:,\s*[^,{}]*)*\s*\}',
        # Nested curly braces: {{X}}
        r'\{\s*\{\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}\s*\}',
        # Curly braces with label: {product: X}
        r'\{\s*(?:product|result|answer|multiplication)(?:\s*(?:is|:|=)\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last match as it's likely the final answer
            return matches[-1].strip()
    
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
        number_match = re.search(r'^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$', text.strip())
        if number_match:
            number_str = number_match.group(0)
            # Convert to int or float as appropriate
            try:
                return int(float(number_str)) if '.' not in number_str and 'e' not in number_str.lower() else float(number_str)
            except ValueError:
                return None
    
    # Look for a number in square brackets with clear product labeling
    bracket_product_match = re.search(r'\[(?:product|result|answer|multiplication)(?:\s*(?:is|=)\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\]', text, re.IGNORECASE)
    if bracket_product_match:
        number_str = bracket_product_match.group(1)
        # Convert to int or float as appropriate
        try:
            return int(float(number_str)) if '.' not in number_str and 'e' not in number_str.lower() else float(number_str)
        except ValueError:
            return None
    
    # Look for a number in parentheses with clear product labeling
    paren_product_match = re.search(r'\((?:product|result|answer|multiplication)(?:\s*(?:is|=)\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)', text, re.IGNORECASE)
    if paren_product_match:
        number_str = paren_product_match.group(1)
        # Convert to int or float as appropriate
        try:
            return int(float(number_str)) if '.' not in number_str and 'e' not in number_str.lower() else float(number_str)
        except ValueError:
            return None
    
    # Look for a number preceded by "product" or "multiplication" with clear labeling
    product_match = re.search(r'(?:product|multiplication|result|answer|multiply)(?:\s+(?:of|the|numbers|value|result))?\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text.lower())
    if product_match:
        number_str = product_match.group(1)
        # Convert to int or float as appropriate
        try:
            return int(float(number_str)) if '.' not in number_str and 'e' not in number_str.lower() else float(number_str)
        except ValueError:
            return None
    
    # Look for a number at the end of the text (often the final answer)
    end_number_match = re.search(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)$', text.strip())
    if end_number_match:
        number_str = re.sub(r'(\d),(\d)', r'\1\2', end_number_match.group(1))
        try:
            value = float(number_str)
            return int(value) if value.is_integer() else value
        except ValueError:
            pass
    
    # Look for multiplication steps that lead to a final result
    mult_steps_match = re.search(r'(\d+)\s*[×*]\s*(\d+)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$', text)
    if mult_steps_match:
        result = mult_steps_match.group(3)
        try:
            value = float(result)
            return int(value) if value.is_integer() else value
        except ValueError:
            pass
    
    # Look for a number in a list context that might be the product
    list_product_match = re.search(r'\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]', text)
    if list_product_match:
        number_str = list_product_match.group(1)
        try:
            value = float(number_str)
            return int(value) if value.is_integer() else value
        except ValueError:
            pass
    
    # Look for a standalone number in the text (as a last resort)
    number_matches = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
    if number_matches:
        # Remove commas from numbers
        cleaned_matches = [re.sub(r'(\d),(\d)', r'\1\2', match) for match in number_matches]
        
        # Try to find the most likely answer (prefer larger numbers as they're more likely to be the product)
        try:
            # Convert all matches to floats for comparison
            float_matches = [float(match) for match in cleaned_matches]
            
            # If there's only one number, return it
            if len(float_matches) == 1:
                value = float_matches[0]
                return int(value) if value.is_integer() else value
            
            # If there are multiple numbers, prefer the largest absolute value
            # (products are typically larger than the individual factors)
            abs_values = [(i, abs(val)) for i, val in enumerate(float_matches)]
            abs_values.sort(key=lambda x: x[1], reverse=True)
            
            # Return the largest absolute value
            largest_idx = abs_values[0][0]
            value = float_matches[largest_idx]
            return int(value) if value.is_integer() else value
        except ValueError:
            pass
    
    # Look for "answer" or "product" followed by a number
    answer_match = re.search(r'(?:answer|product|result)(?:\s*(?:is|:|=)\s*)([+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)', text, re.IGNORECASE)
    if answer_match:
        number_str = re.sub(r'(\d),(\d)', r'\1\2', answer_match.group(1))
        try:
            value = float(number_str)
            return int(value) if value.is_integer() else value
        except ValueError:
            pass
    
    # Look for a number after "So," or "Therefore," which often indicates the final answer
    conclusion_match = re.search(r'(?:so|therefore|thus|hence),?\s+(?:the )?(?:product|result|answer|value)\s+(?:is|=)\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)', text.lower())
    if conclusion_match:
        number_str = re.sub(r'(\d),(\d)', r'\1\2', conclusion_match.group(1))
        try:
            value = float(number_str)
            return int(value) if value.is_integer() else value
        except ValueError:
            pass
    
    # Look for a number in a sentence that ends with a period (likely the conclusion)
    sentence_match = re.search(r'(?:the )?(?:product|result|answer|value)\s+(?:is|=)\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\.', text.lower())
    if sentence_match:
        number_str = re.sub(r'(\d),(\d)', r'\1\2', sentence_match.group(1))
        try:
            value = float(number_str)
            return int(value) if value.is_integer() else value
        except ValueError:
            pass
    
    return None
