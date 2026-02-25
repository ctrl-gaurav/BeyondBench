import re

def extract_from_last_line(text):
    """
    Extract answers from the last line or sentence.

    This function looks for potential answers in the last line or sentence of the text.

    Args:
        text (str): The text to search for answers in the last line or sentence.
        
    Returns:
        The extracted answer or None.
    """
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Regex to capture:
    # - Numbers with or without commas (e.g., 1494 or 1,494)
    # - Quoted text at the end of a line
    
    # number_pattern = r'([+-]?\d+(?:,\d{3})*)'
    
    number_pattern = r'([+-]?\d+(?:,\d{3})*)(?:\.\d+)?'
    quoted_text_pattern = r'[\"\']([^\"\'.]+)[\"\']'
    combined_pattern = fr'{quoted_text_pattern}\.?$|{number_pattern}\.?$'

    # Check the last few lines for potential answers
    for i in range(min(3, len(lines))):
        line = lines[-(i+1)].strip()
        if line:
            answer_match = re.search(combined_pattern, line)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()
    
    # Split the text into sentences
    sentences = re.split(r'[.!?]', text)
    
    # Check the last few sentences for potential answers
    for i in range(min(3, len(sentences))):
        sentence = sentences[-(i+1)].strip()
        if sentence:
            answer_match = re.search(combined_pattern, sentence)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()
    
    sentence = sentences[-1].split("=")
    # number_pattern = r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b'
    match = re.search(number_pattern, sentence[-1])
    return match.group(0) if match else None

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
        r'(?:^|\n|\s)(?:answer|the answer is|final answer|the final answer is)[:\s]+(.+?)\s*\n*\s*(?:\n|$|\.)',
        # The result is X
        r'(?:^|\n|\s)(?:the result is)[:\s]+(.+?)\s*\n*\s*(?:\n|$|\.)',
        # The value is X
        r'(?:^|\n|\s)(?:the value is)[:\s]+(.+?)\s*\n*\s*(?:\n|$|\.)',
        
        r'(?:^|\n|\s)(?:odd numbers in the list is)[:\s]+(.+?)\s*\n*\s*(?:\n|$|\.)',
        # The number is X
        r'(?:^|\n|\s)(?:the number is)[:\s]+(.+?)\s*\n*\s*(?:\n|$|\.)',
        # The output is X
        r'(?:^|\n|\s)(?:the output is)[:\s]+(.+?)\s*\n*\s*(?:\n|$|\.)',
        # The solution is X
        r'(?:^|\n|\s)(?:the solution is)[:\s]+(.+?)\s*\n*\s*(?:\n|$|\.)',
        r'(?:^|\n|\s)(?:the )?(?:total )?(?:count|result|output|answer|value|number)(?: of [^.]*)?\s+is\s+(.+?)\s*\n*\s*(?:\n|$|\.)',
        # X is the answer
        r'(?:^|\n|\s)(.+?)\s*is the answer\b',  ### why does pattern here differ from above
        # X odd numbers
        r'(?:^|\n|\s)(.+?)\s*odd numbers\b',
        # X is the result
        r'(?:^|\n|\s)(.+?)\s*is the result\b',
        # X is the value
        r'(?:^|\n|\s)(.+?)\s*is the value\b',
        #X odd numbers
        r'(?:^|\n|\s)(.+?)\s*odd numbers',
        # X is the number
        r'(?:^|\n|\s)(.+?)\s*is the number\b',
        # X is the output
        r'(?:^|\n|\s)(.+?)\s*is the output\b',
        # X is the solution
        r'(?:^|\n|\s)(.+?)\s*is the solution\b',
        r'(?:^|\n|\s)(.+?)\s*is the count\b',
        # r'(?:^|\n|\s)(?:odd numbers in the list are)[:\s]+(.+?)\s*(?:\n|$|\.)'
        # r'(?:^|\n|\s)([-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\n|$|\.)',
    ]
    
    for index,pattern in enumerate(patterns):
        matches = re.findall(pattern, text.lower())
        if matches:
            # Extract the actual match content (might be a tuple from regex groups)
            match = matches[-1]
            if isinstance(match, tuple):
                match = match[0]
            
            # Clean up the match
            cleaned_match = match.strip()
            # Return the last match as it's likely the final answer
            return cleaned_match, False, index <= 7 # index <= 7 means regex is later for number
    
    return None, False, None

def clean_and_convert_to_number(text,later=None):
    """
    Clean the text and convert it to a number if possible.
    
    Args:
        text (str): The text to clean and convert
        
    Returns:
        int, float, or str: The converted number or the original text if conversion fails
    """
    if not text:
        return None
    
    # Remove LaTeX formatting
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'[*_~`]', '', text)
    
    # Better pattern: supports plain and comma-formatted numbers
    number_pattern = r'([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)'
    number_match = re.findall(number_pattern, text)
    if len(number_match) > 1:
        if later:
            number_match = number_match[0]
        else:
            number_match = number_match[-1]
    elif len(number_match) == 0:
        return None
    else:
        number_match = number_match[0]
    if number_match:
        # number_str = number_match.group(1)
        number_str = number_match.replace(',', '')
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
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
    
    # Look for alternative boxed formats that LLMs might use
    alt_boxed1 = re.findall(r'\\box\{([^{}]*)\}', text)
    alt_boxed2 = re.findall(r'\[\\boxed\{([^{}]*)\}\]', text)
    alt_boxed3 = re.findall(r'\(\boxed\{([^{}]*)\}\)', text)
    
    text_bf = re.findall(r"\\boxed\{\\textbf\{([^{}]*)\}\}",text)
    # Combine all matches
    all_matches = standard_boxed + paren_boxed + bracket_boxed + markdown_boxed + alt_boxed1 + alt_boxed2 + alt_boxed3 + text_bf
    
    if all_matches:
        # Return the last match as it's likely the final answer
        return all_matches[-1].strip(), True
    
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
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Look for answers in the code block
            for match in matches:
                # Look for print statements or return statements
                print_match = re.search(r'print\s*\(\s*["\']?([^"\']+)["\']?\s*\)', match)
                if print_match:
                    return print_match.group(1).strip()
                
                return_match = re.search(r'return\s+([^\s;]+)', match)
                if return_match:
                    return return_match.group(1).strip()
                
                # Look for variable assignments
                assign_match = re.search(r'(?:answer|result)\s*=\s*([^\s;]+)', match)
                if assign_match:
                    return assign_match.group(1).strip()
    
    return None

# def extract_from_equals(text):
#     pass

def parse_odd_count_answer(clean_response):
    """
    Extract an answer from the LLM response with robust pattern matching.
    
    This function attempts to extract the answer from various formats that LLMs might use,
    prioritizing the requested format (boxed) and falling back to other common patterns.
    
    Args:
        clean_response (str): The cleaned response from the LLM
        
    Returns:
        tuple: (bool, value) where bool indicates if instruction was followed
               and value is the extracted answer or None
    """
    # 1. Try to extract from boxed formats (highest priority)
    boxed_answer, instruction_followed = extract_from_boxed_formats(clean_response)
    if boxed_answer is not None:
        answer = clean_and_convert_to_number(boxed_answer)
        return instruction_followed, answer
    
    # 2. Try to extract from explicit answer statements
    explicit_answer, instruction_followed,later = extract_from_explicit_statements(clean_response)
    if explicit_answer is not None:
        answer = clean_and_convert_to_number(explicit_answer,later)
        return instruction_followed, answer
    
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
    
    # 6. No valid answer found
    return False, None
##################################
