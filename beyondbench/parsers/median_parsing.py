import re
import logging

def parse_median_answer(response):
    """
    Extract median value from model response with enhanced robustness for various formats.
    
    Args:
        response (str): The model's response text
        
    Returns:
        float or None: The parsed median value or None if no valid answer is found
    """
    # 1. Try to extract from boxed formats (highest priority)
    boxed_answer, instruction_followed = extract_from_boxed_formats(response)
    if boxed_answer is not None:
        answer = clean_and_convert_to_number(boxed_answer)
        if is_valid_number(answer):
            return answer
    
    # 2. Try to extract from explicit statements
    explicit_answer = extract_from_explicit_statements(response)
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
    
    # 5. Try to extract from median calculation steps
    calc_answer = extract_from_calculation_steps(response)
    if calc_answer is not None:
        answer = clean_and_convert_to_number(calc_answer)
        if is_valid_number(answer):
            return answer
    
    # No valid answer found
    return None


def extract_from_boxed_formats(text):
    """
    Extract answers from various boxed formats.
    
    Args:
        text (str): The text to search for boxed formats
        
    Returns:
        tuple: (extracted_answer, instruction_followed)
    """
    # Replace line breaks with spaces for better regex matching
    text = text.replace('\n', ' ')
    
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
    Extract answers from explicit statements like "Answer: X" or "The median is X".
    
    Args:
        text (str): The text to search for explicit answer statements
        
    Returns:
        The extracted answer or None
    """
    # Common patterns for explicit answer statements related to median calculation
    patterns = [
        # Answer: X or The answer is X
        r'(?:^|\n|\s)(?:answer|the answer is|final answer|the final answer is)[:\s]+([^\n\.;]+)',
        # The median is X
        r'(?:^|\n|\s)(?:the median is|median is|the median value is)[:\s]+([^\n\.;]+)',
        # Therefore, the median is X
        r'(?:^|\n|\s)(?:therefore,? the median is)[:\s]+([^\n\.;]+)',
        # The result is X
        r'(?:^|\n|\s)(?:the result is)[:\s]+([^\n\.;]+)',
        # Value of median: X
        r'(?:^|\n|\s)(?:value of median)[:\s]+([^\n\.;]+)',
        # middle value is X
        r'(?:^|\n|\s)(?:middle value is|the middle value is)[:\s]+([^\n\.;]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # Return the last match as it's likely the final answer
            return matches[-1].strip()
    
    return None


def extract_from_latex_math(text):
    """
    Extract answers from LaTeX math expressions.
    
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


def extract_from_last_line(text):
    """
    Extract answers from the last line or sentence.
    
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
            
            # Look for a simple number at the end of a line
            num_match = re.search(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\.?$', line)
            if num_match:
                return num_match.group(1).strip()
    
    return None


def extract_from_calculation_steps(text):
    """
    Extract the median calculation from working steps.
    
    This function is specialized for median calculations, looking for statements about
    finding the middle value or averaging two middle values.
    
    Args:
        text (str): The text to search for calculation steps
        
    Returns:
        The extracted answer or None
    """
    # Look for even length case - average of two middle values
    # Pattern like "average of X and Y is Z" or "(X + Y) / 2 = Z"
    avg_patterns = [
        r'average of (?:the middle values|the middle numbers)?\s*([+-]?\d+(?:\.\d+)?)\s*and\s*([+-]?\d+(?:\.\d+)?)\s*(?:is|=)\s*([+-]?\d+(?:\.\d+)?)',
        r'\(\s*([+-]?\d+(?:\.\d+)?)\s*\+\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*/\s*2\s*=\s*([+-]?\d+(?:\.\d+)?)',
        r'([+-]?\d+(?:\.\d+)?)\s*\+\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*(?:then|and)\s*([^=]+)\s*=\s*([+-]?\d+(?:\.\d+)?)'
    ]
    
    for pattern in avg_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # For the third pattern, we want the final result (fifth group)
            if len(matches[0]) == 5:
                return matches[-1][4].strip()
            # For the first two patterns, we want the third group
            return matches[-1][2].strip()
    
    # Look for odd length case - middle value
    # Pattern like "middle value is X" or "the middle element is X"
    middle_patterns = [
        r'(?:the )?middle (?:value|element|number) (?:is|=)\s*([+-]?\d+(?:\.\d+)?)',
        r'(?:the )?middle (?:position|index).*?\s*([+-]?\d+(?:\.\d+)?)'
    ]
    
    for pattern in middle_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[-1].strip()
    
    # Look for sorted list and middle index extraction
    sorted_list_pattern = r'sorted (?:list|array).*?:\s*\[([-+\d\s,.]+)\]'
    sorted_match = re.search(sorted_list_pattern, text.lower())
    if sorted_match:
        sorted_list_str = sorted_match.group(1)
        try:
            # Extract numbers from the sorted list string
            nums = [int(n.strip()) for n in re.findall(r'([+-]?\d+)', sorted_list_str)]
            # Extract statements about middle index or position
            middle_idx_pattern = r'(?:middle index|position).*?(\d+)'
            idx_match = re.search(middle_idx_pattern, text.lower())
            if idx_match:
                idx = int(idx_match.group(1))
                # Check if index is valid for the list
                if 0 <= idx < len(nums):
                    return str(nums[idx])
                # Check for averaging two middle values
                if len(nums) % 2 == 0:
                    mid1 = len(nums) // 2 - 1
                    mid2 = len(nums) // 2
                    if 0 <= mid1 < len(nums) and 0 <= mid2 < len(nums):
                        return str((nums[mid1] + nums[mid2]) / 2)
        except (ValueError, IndexError):
            pass
    
    return None


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
    
    # Handle fractions (e.g., 3/4)
    fraction_match = re.search(r'^([+-]?\d+)\s*/\s*(\d+)$', text.strip())
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator != 0:  # Avoid division by zero
            return numerator / denominator
    
    # Handle median calculation as average of two values
    # Look for patterns like "(a + b) / 2 = result"
    avg_match = re.search(r'\(\s*([+-]?\d+(?:\.\d+)?)\s*\+\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*/\s*2\s*=\s*([+-]?\d+(?:\.\d+)?)', text)
    if avg_match:
        return float(avg_match.group(3))
    
    # Handle cases where the model provides a calculation with multiple steps
    # Look for the last "=" in the text
    last_equal_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$', text)
    if last_equal_match:
        result = float(last_equal_match.group(1))
        # Preserve the sign and integer/float type
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