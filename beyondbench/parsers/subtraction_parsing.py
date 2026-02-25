import re

def parse_subtraction_answer(clean_response, prompt=None):
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
            # Check if the original boxed answer had a negative sign
            if isinstance(boxed_answer, str) and boxed_answer.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in the boxed format in the response
            elif isinstance(answer, (int, float)) and answer > 0:
                if "\\boxed{-" in clean_response or "\\boxed{\\text{-}" in clean_response or "\\boxed{\\textbf{-}" in clean_response:
                    answer = -answer
                # Look for negative signs in calculation steps
                elif re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return instruction_followed, answer
    
    # 2. Try to extract from markdown formatting (bold, italic)
    markdown_answer = extract_from_markdown_formatting(clean_response)
    if markdown_answer is not None:
        answer = clean_and_convert_to_number(markdown_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            # Check if the original markdown answer had a negative sign
            if isinstance(markdown_answer, str) and markdown_answer.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in calculation steps
            elif isinstance(answer, (int, float)) and answer > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return False, answer
    
    # 3. Try to extract from explicit answer statements
    explicit_answer, instruction_followed = extract_from_explicit_statements(clean_response)
    if explicit_answer is not None:
        answer = clean_and_convert_to_number(explicit_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            # Check if the original explicit answer had a negative sign
            if isinstance(explicit_answer, str) and explicit_answer.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in calculation steps
            elif isinstance(answer, (int, float)) and answer > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return instruction_followed, answer
    
    # 4. Try to extract from LaTeX math expressions
    latex_answer = extract_from_latex_math(clean_response)
    if latex_answer is not None:
        answer = clean_and_convert_to_number(latex_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            # Check if the original LaTeX answer had a negative sign
            if isinstance(latex_answer, str) and latex_answer.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in calculation steps
            elif isinstance(answer, (int, float)) and answer > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return False, answer
    
    # 5. Try to extract from code blocks
    code_answer = extract_from_code_blocks(clean_response)
    if code_answer is not None:
        answer = clean_and_convert_to_number(code_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            # Check if the original code answer had a negative sign
            if isinstance(code_answer, str) and code_answer.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in calculation steps
            elif isinstance(answer, (int, float)) and answer > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return False, answer
    
    # 6. Try to extract from the last line or sentence
    last_line_answer = extract_from_last_line(clean_response)
    if last_line_answer is not None:
        answer = clean_and_convert_to_number(last_line_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            # Check if the original last line answer had a negative sign
            if isinstance(last_line_answer, str) and last_line_answer.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in calculation steps
            elif isinstance(answer, (int, float)) and answer > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return False, answer
    
    # 7. Try to extract from scientific notation
    scientific_notation = extract_from_scientific_notation(clean_response)
    if scientific_notation is not None:
        answer = clean_and_convert_to_number(scientific_notation)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            # Check if the original scientific notation had a negative sign
            if isinstance(scientific_notation, str) and scientific_notation.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in calculation steps
            elif isinstance(answer, (int, float)) and answer > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return False, answer
    
    # 8. Try to extract from array or list formats
    array_answer = extract_from_array_format(clean_response)
    if array_answer is not None:
        answer = clean_and_convert_to_number(array_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            # Check if the original array answer had a negative sign
            if isinstance(array_answer, str) and array_answer.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in calculation steps
            elif isinstance(answer, (int, float)) and answer > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return False, answer
    
    # 9. Try to extract from calculation steps
    calculation_answer = extract_from_calculation_steps(clean_response, input_numbers)
    if calculation_answer is not None:
        answer = clean_and_convert_to_number(calculation_answer)
        if is_valid_number(answer) and not is_input_number(answer, input_numbers):
            # Check if the original calculation answer had a negative sign
            if isinstance(calculation_answer, str) and calculation_answer.strip().startswith('-') and isinstance(answer, (int, float)) and answer > 0:
                answer = -answer
            # Check for negative signs in calculation steps
            elif isinstance(answer, (int, float)) and answer > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    answer = -answer
            return False, answer
    
    # 10. Try to extract plain numbers from very short responses (lowest priority)
    # Only use this if we couldn't find any other answer and the response is very short
    if len(clean_response.strip()) < 10:
        plain_number = extract_plain_number(clean_response)
        if plain_number is not None and is_valid_number(plain_number) and not is_input_number(plain_number, input_numbers):
            # Check for negative signs in calculation steps
            if isinstance(plain_number, (int, float)) and plain_number > 0:
                if re.search(r'[-]\d+\s*[+]\s*\d+\s*=\s*[-]\d+', clean_response) or re.search(r'[-]\d+\s*[-]\s*\d+\s*=\s*[-]\d+', clean_response):
                    plain_number = -plain_number
            return False, plain_number
    
    # No valid answer found
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
    # Look for subtraction steps with a clear final result
    # Pattern: "X - Y = Z" followed by "Z - W = V" etc.
    steps = re.findall(r'([+-]?\d+(?:\.\d+)?)\s*[-]\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)', text)
    
    if steps:
        # Get the last step's result (likely the final answer)
        last_result = steps[-1][2]
        
        # Check if this is not just an input number
        if not is_input_number(last_result, input_numbers):
            return last_result
    
    # Look for a sequence of calculations with intermediate results
    # Pattern: "First: X - Y = Z", "Then: Z - W = V", etc.
    step_pattern = r'(?:first|then|next|finally|step|lastly|after that)[:\s]*(?:[^=]*=\s*)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    step_matches = re.findall(step_pattern, text.lower())
    
    if step_matches and len(step_matches) > 0:
        # Get the last step's result
        last_result = step_matches[-1]
        
        # Check if this is not just an input number
        if not is_input_number(last_result, input_numbers):
            return last_result
    
    # Look for a clear final calculation step
    final_step = re.search(r'(?:final|result|answer|difference)[:\s]*(?:[^=]*=\s*)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text.lower())
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
    
    # Handle cases where the model provides a calculation like "X - Y = Z"
    calc_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*[-]\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)', text)
    if calc_match:
        result = float(calc_match.group(3))
        # Preserve the sign
        if '.' not in calc_match.group(3) and 'e' not in calc_match.group(3).lower():
            return int(result)
        else:
            return result
    
    # Handle cases where the model provides a calculation with multiple steps
    # Look for the last "=" in the text
    last_equal_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$', text)
    if last_equal_match:
        result = float(last_equal_match.group(1))
        # Preserve the sign
        if '.' not in last_equal_match.group(1) and 'e' not in last_equal_match.group(1).lower():
            return int(result)
        else:
            return result
    
    # Try to convert to a number
    try:
        # Extract the first number from the text, including scientific notation
        number_match = re.search(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text)
        if number_match:
            number_str = number_match.group(1)
            # Convert to float first to preserve the sign, then to int if appropriate
            result = float(number_str)
            if '.' not in number_str and 'e' not in number_str.lower():
                return int(result)
            else:
                return result
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
    
    # Look for LaTeX boxed format with parentheses: \(\boxed{answer}\)
    paren_boxed = re.findall(r'\\[\(\[]\\boxed\{([^{}]*)\}\\[\)\]]', text)
    
    # Look for LaTeX boxed format with brackets: \[\boxed{answer}\]
    bracket_boxed = re.findall(r'\\[\[\]]\\boxed\{([^{}]*)\}\\[\[\]]', text)
    
    # Look for markdown-style boxed format: [boxed{answer}]
    markdown_boxed = re.findall(r'\[boxed\{([^{}]*)\}\]', text)
    
    # Look for formats with negative signs in different positions
    neg_boxed = re.findall(r'\\boxed\{-\s*([^{}]*)\}', text)
    neg_boxed2 = re.findall(r'\\boxed\{\\text\{-\}\s*([^{}]*)\}', text)
    neg_boxed3 = re.findall(r'\\boxed\{\\textbf\{-\}\s*([^{}]*)\}', text)
    
    # Combine all matches
    all_matches = standard_boxed + paren_boxed + bracket_boxed + markdown_boxed
    
    # Process negative boxed formats
    for match in neg_boxed + neg_boxed2 + neg_boxed3:
        try:
            value = "-" + match
            all_matches.append(value)
        except (ValueError, TypeError):
            pass
    
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
    # Common patterns for explicit answer statements
    patterns = [
        # Answer: X or The answer is X
        r'(?:^|\n|\s)(?:answer|the answer is|final answer|the final answer is)[:\s]+([^\n\.]+)',
        # Therefore, the answer is X
        r'(?:^|\n|\s)(?:therefore,? the answer is)[:\s]+([^\n\.]+)',
        # The result is X
        r'(?:^|\n|\s)(?:the result is)[:\s]+([^\n\.]+)',
        # The difference is X
        r'(?:^|\n|\s)(?:the difference is)[:\s]+([^\n\.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # Return the last match as it's likely the final answer
            return matches[-1].strip(), False
    
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
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last match as it's likely the final answer
            return matches[-1].strip()
    
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
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Check the last few lines for potential answers
    for i in range(min(3, len(lines))):
        line = lines[-(i+1)].strip()
        if line:
            # Look for patterns like "X" or "X." at the end of the line
            answer_match = re.search(r'[\"\']([^\"\'.]+)[\"\']\.?$|([+-]?\d+(?:\.\d+)?)\.?$', line)
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
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last match as it's likely the final answer
            if isinstance(matches[-1], tuple):
                base, exponent = matches[-1]
                try:
                    value = float(base) * (10 ** int(exponent))
                    return str(value)
                except (ValueError, TypeError):
                    pass
            else:
                return matches[-1].strip()
    
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
        # Parentheses with single element: (X)
        r'\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)',
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
    
    return None
