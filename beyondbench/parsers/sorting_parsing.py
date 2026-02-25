import re
import json
import logging

def parse_sorted_list(response):
    """
    Extract and parse a sorted list from a model response with robust pattern matching.
    
    Args:
        response (str): The model's response text
        
    Returns:
        list or None: Parsed list of integers if found, None otherwise
    """
    # Clean response for consistent processing
    clean_response = response.replace('\\n', ' ').strip()
    
    # Try multiple extraction methods in order of priority
    extracted_list = (
        extract_from_boxed_format(clean_response) or
        extract_from_list_formats(clean_response) or
        extract_from_explicit_statements(clean_response) or
        extract_from_code_blocks(clean_response) or
        extract_from_final_line(clean_response)
    )
    
    return extracted_list

def extract_from_boxed_format(text):
    """Extract sorted list from various boxed formats."""
    # Find all potential boxed answers with flexible parsing
    patterns = [
        # Standard LaTeX boxed format
        r'\\boxed{([^{}]+)}',
        # LaTeX boxed with text command
        r'\\boxed{\\text{([^{}]+)}}',
        # LaTeX boxed with math environment
        r'\\[\(\[]\\boxed{([^{}]+)}\\[\)\]]',
        # Markdown-style boxed
        r'\[boxed{([^{}]+)}\]',
        # LaTeX boxed with array command
        r'\\boxed{\\begin{array}{[^}]*}([^}]+)\\end{array}}',
        # Nested boxed
        r'\\boxed{\\boxed{([^{}]+)}}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):  # Try from the last match
                cleaned_list = parse_number_list(match)
                if cleaned_list:
                    return cleaned_list
    
    return None

def extract_from_list_formats(text):
    """Extract sorted list from various list notation formats."""
    patterns = [
        # Standard array notation [a, b, c]
        r'\[([^\[\]]+)\]',
        # Parentheses notation (a, b, c)
        r'\(([^()]+)\)',
        # LaTeX array notation \{a, b, c\}
        r'\\{([^{}]+)\\}',
        # LaTeX array brackets \begin{bmatrix} a \\ b \\ c \end{bmatrix}
        r'\\begin{[a-z]*matrix}([^}]+)\\end{[a-z]*matrix}',
        # List keyword followed by items
        r'(?:list|sorted list|sorted array)[:=]\s*(?:\[|\{|\()([^]})]+)(?:\]|\}|\))'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                cleaned_list = parse_number_list(match)
                if cleaned_list:
                    return cleaned_list
    
    return None

def extract_from_explicit_statements(text):
    """Extract sorted list from explicit statements in the text."""
    patterns = [
        # "The sorted list is a, b, c"
        r'(?:the |my |final )?(?:sorted|ordered|arranged|resulting) (?:list|array|sequence|numbers) (?:is|would be|will be|becomes)[:\s]+([^\.]+)',
        # "After sorting, we get a, b, c"
        r'(?:after sorting|sorting gives|sorted version)[:\s]+([^\.]+)',
        # "Therefore, the sorted list is a, b, c"
        r'(?:therefore|thus|hence|so|consequently)[,\s]+(?:the )?(?:sorted|ordered|arranged) (?:list|array|sequence|numbers) (?:is|are|would be|will be)[:\s]+([^\.]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            for match in reversed(matches):
                cleaned_list = parse_number_list(match)
                if cleaned_list:
                    return cleaned_list
    
    return None

def extract_from_code_blocks(text):
    """Extract sorted list from code blocks in the text."""
    # Find code blocks: ```X```, `X`
    patterns = [
        r'```(?:python|plaintext)?\s*\n([^`]+)\n```',
        r'`([^`]+)`',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                # Look for list-like structures within code
                code_list_patterns = [
                    r'\[([^\[\]]+)\]',  # [a, b, c]
                    r'=\s*\[([^\[\]]+)\]',  # = [a, b, c]
                    r'sorted_list\s*=\s*\[([^\[\]]+)\]',  # sorted_list = [a, b, c]
                    r'(?:array|list)(?:\(\s*)?(?:\[)?([^)]+)(?:\])?(?:\s*\))?'  # array([a, b, c]) or list[a, b, c]
                ]
                
                for code_pattern in code_list_patterns:
                    code_matches = re.findall(code_pattern, match)
                    if code_matches:
                        for code_match in reversed(code_matches):
                            cleaned_list = parse_number_list(code_match)
                            if cleaned_list:
                                return cleaned_list
                
                # Try parsing the entire code block as a list
                cleaned_list = parse_number_list(match)
                if cleaned_list:
                    return cleaned_list
    
    return None

def extract_from_final_line(text):
    """Extract sorted list from the final line of the text."""
    # Split into lines and check the last few
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return None
    
    # Check the last 3 lines for list patterns
    for i in range(min(3, len(lines))):
        line = lines[-(i+1)]
        
        # Look for list patterns in the line
        list_patterns = [
            r'\[([^\[\]]+)\]',  # [a, b, c]
            r'\(([^()]+)\)',    # (a, b, c)
            r'(?:^|\s)(-?\d+(?:,\s*-?\d+)+)'  # a, b, c
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, line)
            if matches:
                for match in matches:
                    cleaned_list = parse_number_list(match)
                    if cleaned_list:
                        return cleaned_list
    
    return None

def parse_number_list(text):
    """
    Parse a list of numbers from various text formats.
    
    Args:
        text (str or list): Text or list containing numbers
        
    Returns:
        list: List of integers if parsing successful, None otherwise
    """
    if not text:
        return None
    
    # If already a list, try to convert elements to integers
    if isinstance(text, list):
        try:
            return [int(float(x)) if isinstance(x, (str, int, float)) else None for x in text]
        except (ValueError, TypeError):
            pass
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None
    
    # Clean the text
    text = text.strip()
    
    # Try to parse as JSON if it looks like JSON
    if (text.startswith('[') and text.endswith(']')) or (text.startswith('{') and text.endswith('}')):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                try:
                    return [int(float(x)) if isinstance(x, (str, int, float)) else None for x in parsed]
                except (ValueError, TypeError):
                    pass
        except json.JSONDecodeError:
            pass
    
    # Try to handle LaTeX array notation
    text = re.sub(r'\\\\', ',', text)  # Replace LaTeX line breaks with commas
    
    # Handle LaTeX formatting
    text = re.sub(r'\\[a-zA-Z]+(?:{[^{}]*})?', '', text)  # Remove LaTeX commands
    
    # Look for comma-separated values or space-separated values
    # First try comma-separated
    numbers = []
    try:
        # Split by commas and process each part
        parts = re.split(r'[,\s]+', text)
        for part in parts:
            part = part.strip()
            if part:
                # Handle numbers with thousand separators
                part = part.replace(',', '')
                num = int(float(part))
                numbers.append(num)
        
        if numbers:
            return numbers
    except (ValueError, TypeError):
        pass
    
    # Try extracting any numbers in order
    try:
        numbers = [int(float(x)) for x in re.findall(r'-?\d+(?:\.\d+)?', text)]
        if numbers:
            return numbers
    except (ValueError, TypeError):
        pass
    
    return None