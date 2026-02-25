import re
import logging

def parse_comparison_result(response):
    """
    Extract comparison result from a model response with robust pattern matching.
    
    Args:
        response (str): The model's response text
        
    Returns:
        str or None: Normalized comparison result ('greater than', 'less than', 'equal to') 
                     if found, None otherwise
    """
    # Clean response for consistent processing
    clean_response = response.replace('\n', ' ').strip()
    
    # Try multiple extraction methods in order of priority
    extracted_result = (
        extract_from_boxed_format(clean_response) or
        extract_from_explicit_statement(clean_response) or
        extract_from_final_sentence(clean_response) or
        extract_from_comparison_symbols(clean_response)
    )
    
    if extracted_result:
        return normalize_comparison_result(extracted_result)
    
    return None

def extract_from_boxed_format(text):
    """Extract comparison result from various boxed formats."""
    # Standard LaTeX boxed format
    patterns = [
        # Basic LaTeX boxed format
        r'\\boxed{([^{}]+)}',
        # LaTeX boxed with text command
        r'\\boxed{\\text{([^{}]+)}}',
        # LaTeX boxed with textbf command
        r'\\boxed{\\textbf{([^{}]+)}}',
        # LaTeX math environment with boxed
        r'\\[\(\[]\\boxed{([^{}]+)}\\[\)\]]',
        # Markdown-style boxed
        r'\[boxed{([^{}]+)}\]',
        # LaTeX boxed with math command
        r'\\boxed{\$([^$]+)\$}',
        # Double boxed (nested)
        r'\\boxed{\\boxed{([^{}]+)}}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last match as it's likely the final answer
            return matches[-1].strip()
    
    return None

def extract_from_explicit_statement(text):
    """Extract comparison result from explicit statements in the text."""
    patterns = [
        # "The answer is X" or "The relationship is X"
        r'(?:the |final |my )?(?:answer|result|relationship|relation|conclusion) (?:is|would be|will be|comes out as)[:\s]+([^\.]+)',
        # "Therefore, X" statement
        r'(?:therefore|thus|hence|so|consequently|as a result)[,\s]+(?:the )?(?:answer|result|relationship|relation|conclusion) (?:is|would be|will be)[:\s]+([^\.]+)',
        # "Number 1 is X Number 2"
        r'(?:number|num)[\s\.]*1[\s\.]* (?:is|appears to be|seems to be|would be|will be) ([^\.]+) (?:than )?(?:number|num)[\s\.]*2'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[-1].strip()
    
    return None

def extract_from_final_sentence(text):
    """Extract comparison result from the final sentence."""
    # Split into sentences and check the last few
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return None
    
    # Check the last 2 sentences for comparison terms
    for i in range(min(2, len(sentences))):
        sentence = sentences[-(i+1)].lower()
        
        # Look for comparison indicators
        if any(term in sentence for term in ['greater than', 'less than', 'equal to', '>', '<', '=']):
            # Extract the relation part
            for relation in ['greater than', 'less than', 'equal to']:
                if relation in sentence:
                    return relation
            
            # Check for symbols
            if '>' in sentence:
                return 'greater than'
            elif '<' in sentence:
                return 'less than'
            elif '=' in sentence:
                return 'equal to'
    
    return None

def extract_from_comparison_symbols(text):
    """Extract comparison result from symbolic comparison statements."""
    # Look for patterns like "X > Y", "X < Y", "X = Y"
    match = re.search(r'(\d+)\s*(>|<|=)\s*(\d+)', text)
    if match:
        symbol = match.group(2)
        num1 = int(match.group(1))
        num2 = int(match.group(3))
        
        # Extract the number mention pattern to determine which is Number 1 and Number 2
        num1_pattern = re.search(r'number\s*1\s*(?::|is|=)\s*(\d+)', text.lower())
        num2_pattern = re.search(r'number\s*2\s*(?::|is|=)\s*(\d+)', text.lower())
        
        # Determine which number is Number 1 based on context
        if num1_pattern and num2_pattern:
            num1_value = int(num1_pattern.group(1))
            num2_value = int(num2_pattern.group(1))
            
            # Reverse the comparison if needed based on which number is Number 1
            if num1 == num2_value and num2 == num1_value:
                symbol = '>' if symbol == '<' else ('<' if symbol == '>' else '=')
        
        if symbol == '>':
            return 'greater than'
        elif symbol == '<':
            return 'less than'
        elif symbol == '=':
            return 'equal to'
    
    return None

def normalize_comparison_result(result):
    """Normalize various comparison phrasings to standard format."""
    result = result.lower().strip()
    
    # Check for greater than variations
    if any(term in result for term in ['greater', '>', 'more', 'larger', 'bigger', 'exceeds']):
        return 'greater than'
    
    # Check for less than variations
    elif any(term in result for term in ['less', '<', 'smaller', 'lower', 'below']):
        return 'less than'
    
    # Check for equal to variations
    elif any(term in result for term in ['equal', '=', 'same', 'identical']):
        return 'equal to'
    
    # Handle numbered format (e.g., "Number 1 is less than Number 2")
    elif re.search(r'number\s*1.*number\s*2', result):
        if any(term in result for term in ['greater', '>', 'more', 'larger', 'bigger']):
            return 'greater than'
        elif any(term in result for term in ['less', '<', 'smaller', 'lower']):
            return 'less than'
        elif any(term in result for term in ['equal', '=', 'same']):
            return 'equal to'
    
    # Handle reversed format (e.g., "Number 2 is less than Number 1")
    elif re.search(r'number\s*2.*number\s*1', result):
        if any(term in result for term in ['greater', '>', 'more', 'larger', 'bigger']):
            return 'less than'  # Reversed
        elif any(term in result for term in ['less', '<', 'smaller', 'lower']):
            return 'greater than'  # Reversed
        elif any(term in result for term in ['equal', '=', 'same']):
            return 'equal to'  # Same both ways
    
    return None