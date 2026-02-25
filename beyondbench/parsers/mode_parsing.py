import re
import logging
from collections import Counter

def parse_mode_answer(response):
    """
    Extract mode value(s) from model response with robust pattern matching.
    
    This function attempts to extract mode values from various formats that LLMs might use,
    prioritizing the requested boxed format and falling back to other common patterns.
    
    Args:
        response (str): The response from the LLM
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Clean and standardize the response
    clean_response = response.replace('\n', ' ')
    
    # Try different extraction methods in order of priority
    
    # 1. Try to extract from boxed formats (highest priority)
    boxed_modes = extract_from_boxed_formats(clean_response)
    if boxed_modes:
        return boxed_modes
    
    # 2. Try to extract from markdown formatting (bold, italic)
    markdown_modes = extract_from_markdown_formatting(clean_response)
    if markdown_modes:
        return markdown_modes
    
    # 3. Try to extract from explicit answer statements
    explicit_modes = extract_from_explicit_statements(clean_response)
    if explicit_modes:
        return explicit_modes
    
    # 4. Try to extract from LaTeX math expressions
    latex_modes = extract_from_latex_math(clean_response)
    if latex_modes:
        return latex_modes
    
    # 5. Try to extract from code blocks
    code_modes = extract_from_code_blocks(clean_response)
    if code_modes:
        return code_modes
    
    # 6. Try to extract from array or list formats
    array_modes = extract_from_array_format(clean_response)
    if array_modes:
        return array_modes
    
    # 7. Try to extract from the last line or sentence
    last_line_modes = extract_from_last_line(clean_response)
    if last_line_modes:
        return last_line_modes
    
    # No valid modes found
    return None


def extract_from_boxed_formats(text):
    """
    Extract mode values from various boxed formats.
    
    This function looks for different LaTeX-style boxed formats that LLMs might use.
    
    Args:
        text (str): The text to search for boxed formats
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Look for standard LaTeX boxed format: \boxed{modes}
    standard_boxed = re.search(r'\\boxed\{([^{}]*)\}', text)
    
    # Look for LaTeX boxed format with parentheses: \(\boxed{modes}\)
    paren_boxed = re.search(r'\\[\(\[]\\boxed\{([^{}]*)\}\\[\)\]]', text)
    
    # Look for LaTeX boxed format with brackets: \[\boxed{modes}\]
    bracket_boxed = re.search(r'\\[\[\]]\\boxed\{([^{}]*)\}\\[\[\]]', text)
    
    # Look for markdown-style boxed format: [boxed{modes}]
    markdown_boxed = re.search(r'\[boxed\{([^{}]*)\}\]', text)
    
    # Process the first match found
    match = standard_boxed or paren_boxed or bracket_boxed or markdown_boxed
    
    if match:
        modes_text = match.group(1).strip()
        return parse_modes_from_text(modes_text)
    
    return None


def extract_from_markdown_formatting(text):
    """
    Extract mode values from markdown formatting like bold or italic.
    
    Args:
        text (str): The text to search for markdown formatting
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Look for bold formatting with asterisks: **modes**
    bold_asterisks = re.search(r'\*\*([^*]+)\*\*', text)
    
    # Look for bold formatting with underscores: __modes__
    bold_underscores = re.search(r'__([^_]+)__', text)
    
    # Look for italic formatting with asterisks: *modes*
    italic_asterisks = re.search(r'(?<!\*)\*([^*]+)\*(?!\*)', text)
    
    # Look for italic formatting with underscores: _modes_
    italic_underscores = re.search(r'(?<!_)_([^_]+)_(?!_)', text)
    
    # Process the first match found
    match = bold_asterisks or bold_underscores or italic_asterisks or italic_underscores
    
    if match:
        modes_text = match.group(1).strip()
        return parse_modes_from_text(modes_text)
    
    return None


def extract_from_explicit_statements(text):
    """
    Extract mode values from explicit statements like "Modes: X, Y, Z".
    
    Args:
        text (str): The text to search for explicit statements about modes
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Common patterns for explicit mode statements
    patterns = [
        # Mode: X or The mode is X
        r'(?:^|\n|\s)(?:mode|the mode is|modes are|the modes are)[:\s]+([^\n\.]+)',
        # Therefore, the mode is X
        r'(?:^|\n|\s)(?:therefore,? the mode is|therefore,? the modes are)[:\s]+([^\n\.]+)',
        # The most frequent value(s) is/are X
        r'(?:^|\n|\s)(?:the most frequent values? (?:is|are))[:\s]+([^\n\.]+)',
        # The value(s) that appear(s) most frequently is/are X
        r'(?:^|\n|\s)(?:the values? that appears? most frequently (?:is|are))[:\s]+([^\n\.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            modes_text = match.group(1).strip()
            return parse_modes_from_text(modes_text)
    
    return None


def extract_from_latex_math(text):
    """
    Extract mode values from LaTeX math expressions.
    
    Args:
        text (str): The text to search for LaTeX math expressions
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Look for LaTeX math expressions: $X$, $$X$$, \(X\), \[X\]
    patterns = [
        r'\$([^$]+)\$',
        r'\$\$([^$]+)\$\$',
        r'\\[\(\[]([^\\]+)\\[\)\]]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            modes_text = match.group(1).strip()
            return parse_modes_from_text(modes_text)
    
    return None


def extract_from_code_blocks(text):
    """
    Extract mode values from code blocks.
    
    Args:
        text (str): The text to search for code blocks
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Look for code blocks: ```X```, `X`
    patterns = [
        r'```(?:python|plaintext)?\s*\n([^`]+)\n```',
        r'`([^`]+)`',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            modes_text = match.group(1).strip()
            return parse_modes_from_text(modes_text)
    
    return None


def extract_from_array_format(text):
    """
    Extract mode values from array or list formats.
    
    Args:
        text (str): The text to search for array or list formats
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Look for array or list patterns like [X, Y, Z] or (X, Y, Z)
    patterns = [
        r'\[\s*([^\[\]]*)\s*\]',  # [X, Y, Z]
        r'\(\s*([^\(\)]*)\s*\)',  # (X, Y, Z)
        r'\{\s*([^\{\}]*)\s*\}',  # {X, Y, Z}
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            modes_text = match.group(1).strip()
            return parse_modes_from_text(modes_text)
    
    return None


def extract_from_last_line(text):
    """
    Extract mode values from the last line or sentence.
    
    Args:
        text (str): The text to search for modes in the last line or sentence
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Check the last few lines for potential modes
    for i in range(min(3, len(lines))):
        line = lines[-(i+1)].strip()
        if line:
            # Look for patterns like "X, Y, Z" or "X, Y, and Z" at the end of the line
            modes_match = re.search(r'([0-9,\s]+(?:\s+and\s+[0-9]+)?)\.?$', line)
            if modes_match:
                modes_text = modes_match.group(1).strip()
                return parse_modes_from_text(modes_text)
    
    return None


def parse_modes_from_text(text):
    """
    Parse a string containing comma-separated mode values into a list of integers.
    
    This function handles various formats like "X, Y, Z", "X,Y,Z", "X, Y and Z", etc.
    
    Args:
        text (str): The text containing comma-separated mode values
        
    Returns:
        list: The extracted mode values as a list of integers, or empty list if parsing fails
    """
    if not text:
        return []
    
    # Handle "X, Y, and Z" format
    text = re.sub(r'\s+and\s+', ', ', text)
    text = re.sub(r'\s+or\s+', ', ', text)
    
    # Handle "X and Y" format (without comma)
    text = re.sub(r'(\d+)\s+and\s+(\d+)', r'\1, \2', text)
    text = re.sub(r'(\d+)\s+or\s+(\d+)', r'\1, \2', text)
    
    # Handle various delimiters
    text = re.sub(r'[;|]', ',', text)
    
    # Clean up any text or non-numeric content around the numbers
    # Split by commas and clean each item
    items = []
    for item in text.split(','):
        # Extract any integers from the item
        num_match = re.search(r'(-?\d+)', item)
        if num_match:
            try:
                items.append(int(num_match.group(1)))
            except ValueError:
                pass
    
    # Remove duplicates while preserving order
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    
    return unique_items if unique_items else []


def extract_modes_from_frequency_count(text):
    """
    Extract modes by analyzing frequency counts in the response.
    
    This is a fallback method for when the model provides detailed frequency analysis
    but doesn't clearly state the mode(s).
    
    Args:
        text (str): The text to analyze for frequency counts
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Look for patterns like "1: 3 times, 2: 5 times, 3: 5 times"
    freq_pattern = r'(\d+)(?:\s*:\s*|\s+appears\s+)(\d+)(?:\s*times|\s*occurrences)'
    frequencies = re.findall(freq_pattern, text)
    
    if frequencies:
        # Convert to (value, frequency) pairs
        value_freqs = [(int(value), int(freq)) for value, freq in frequencies]
        
        # Find the maximum frequency
        max_freq = max(freq for _, freq in value_freqs)
        
        # Extract values with the maximum frequency
        modes = [value for value, freq in value_freqs if freq == max_freq]
        
        # Sort for consistency
        modes.sort()
        
        return modes
    
    return None


def extract_from_table_format(text):
    """
    Extract modes from table formats in the response.
    
    This handles cases where the model presents results in a markdown or ASCII table.
    
    Args:
        text (str): The text to analyze for table formats
        
    Returns:
        list or None: The extracted mode values as a list of integers, or None if no valid modes found
    """
    # Look for markdown tables with headers and values
    table_rows = re.findall(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|', text)
    
    if table_rows:
        # Convert to (value, frequency) pairs
        value_freqs = [(int(value), int(freq)) for value, freq in table_rows]
        
        # Find the maximum frequency
        max_freq = max(freq for _, freq in value_freqs)
        
        # Extract values with the maximum frequency
        modes = [value for value, freq in value_freqs if freq == max_freq]
        
        # Sort for consistency
        modes.sort()
        
        return modes
    
    return None