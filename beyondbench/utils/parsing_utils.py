"""
Comprehensive parsing utilities for beyondbench package.

Based on the robust implementation from BeyondBench codebase with enhanced
error handling and multi-strategy answer extraction.
"""

import re
import logging
from typing import Union, Optional, List, Dict, Any
from fractions import Fraction
import ast


class AnswerParser:
    """
    Robust parser for extracting answers from LLM responses.

    Supports multiple answer formats and provides fallback mechanisms
    for reliable answer extraction across different response styles.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the parser.

        Args:
            debug: Enable debug logging for parsing attempts
        """
        self.debug = debug
        self.parsing_attempts = []
        self.logger = logging.getLogger(__name__)

    def parse_answer(
        self,
        response: str,
        expected_type: str = "auto",
        task_type: str = "general"
    ) -> Optional[Union[int, float, str]]:
        """
        Parse answer from LLM response using multiple strategies.

        Args:
            response: The LLM's response text
            expected_type: Expected answer type (int, float, str, auto)
            task_type: Task type for context-aware parsing

        Returns:
            Parsed answer or None if parsing fails
        """
        if not response or not isinstance(response, str):
            return None

        # Reset parsing attempts for debugging
        self.parsing_attempts = []

        # Clean response for consistent processing
        clean_response = self._clean_response(response)

        # Try parsing strategies in priority order
        strategies = [
            self._parse_boxed_format,
            self._parse_explicit_statements,
            self._parse_final_answer_patterns,
            self._parse_mathematical_expressions,
            self._parse_comparison_symbols,
            self._parse_fallback_numbers
        ]

        for strategy in strategies:
            try:
                result = strategy(clean_response, task_type)
                if result is not None:
                    if self.debug:
                        self.logger.debug(f"Successful parsing with {strategy.__name__}: {result}")

                    # Convert to expected type
                    return self._convert_to_expected_type(result, expected_type)

            except Exception as e:
                if self.debug:
                    self.logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue

        if self.debug:
            self.logger.warning(f"All parsing strategies failed for response: {response[:100]}...")

        return None

    def _clean_response(self, response: str) -> str:
        """Clean and normalize response for consistent processing."""
        # Replace various unicode characters and normalize spacing
        cleaned = response.replace('\n', ' ').replace('\r', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Normalize common mathematical symbols
        cleaned = cleaned.replace('×', '*').replace('÷', '/')
        cleaned = cleaned.replace('−', '-').replace('–', '-').replace('—', '-')

        return cleaned

    def _parse_boxed_format(self, text: str, task_type: str = "general") -> Optional[str]:
        """Extract answer from various boxed formats."""
        # Comprehensive boxed format patterns
        patterns = [
            # Standard LaTeX boxed format
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
            r'\\boxed{\\boxed{([^{}]+)}}',
            # Alternative boxed formats
            r'\\box{([^{}]+)}',
            r'\$\\boxed{([^{}]+)}\$',
            # Simple brackets that might be used instead
            r'\[([^\[\]]+)\](?=\s*$)',  # Brackets at end of response
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the last match (most likely to be the final answer)
                answer = matches[-1].strip()
                if answer:
                    return self._clean_extracted_answer(answer)

        return None

    def _parse_explicit_statements(self, text: str, task_type: str = "general") -> Optional[str]:
        """Extract answer from explicit statement patterns."""
        # Comprehensive explicit statement patterns
        patterns = [
            # Standard answer patterns
            r'(?:the |final )?(?:answer|result|solution) (?:is|would be|equals?)[:\s]+([^\.]+)',
            r'(?:therefore|thus|hence)[,\s]+(?:the )?(?:answer|result) (?:is|would be)[:\s]+([^\.]+)',
            r'(?:in conclusion|to conclude)[,\s]+(?:the )?(?:answer|result) (?:is|would be)[:\s]+([^\.]+)',

            # Comparative patterns for comparison tasks
            r'(\d+)\s+(?:is\s+)?(?:greater than|less than|equal to)\s+(\d+)',
            r'(?:the\s+)?(?:first|second)\s+number\s+(?:is\s+)?(?:greater than|less than|equal to)',

            # Mathematical result patterns
            r'(?:equals?|=)\s*([^,\.]+)(?:\.|,|$)',
            r'(?:the\s+)?(?:value|number|count|sum|total|result)\s+(?:is|equals?)\s*([^,\.]+)',

            # Sequence-specific patterns
            r'(?:the\s+)?(?:next|following)\s+(?:term|number|element)\s+(?:is|would be)\s*([^,\.]+)',
            r'(?:continuing\s+)?(?:the\s+)?(?:pattern|sequence)[,\s]+(?:the\s+)?(?:next|following)\s+(?:term|number)\s+(?:is|would be)\s*([^,\.]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Handle different match structures
                if isinstance(matches[0], tuple):
                    # For comparison patterns, process differently
                    return self._process_comparison_match(matches[0], task_type)
                else:
                    answer = matches[-1].strip()
                    if answer:
                        return self._clean_extracted_answer(answer)

        return None

    def _parse_final_answer_patterns(self, text: str, task_type: str = "general") -> Optional[str]:
        """Extract answer from final answer patterns."""
        # Patterns that look for answers near the end
        patterns = [
            r'(?:final answer|answer|result):\s*([^,\.]+?)(?:\.|$)',
            r'(?:the answer is|answer is|result is)\s*([^,\.]+?)(?:\.|$)',
            r'(?:so|therefore|thus),?\s*([^,\.]+?)(?:\.|$)',
        ]

        # Split by sentences and check the last few
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences[-3:]:  # Check last 3 sentences
            for pattern in patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    if answer:
                        return self._clean_extracted_answer(answer)

        return None

    def _parse_mathematical_expressions(self, text: str, task_type: str = "general") -> Optional[str]:
        """Extract answer from mathematical expressions."""
        # Look for expressions that are likely answers
        patterns = [
            # Standalone numbers at end of response
            r'(\b\d+(?:\.\d+)?)\s*$',
            # Fractions
            r'(\d+\/\d+)',
            # Scientific notation
            r'(\d+(?:\.\d+)?[eE][+-]?\d+)',
            # Percentages
            r'(\d+(?:\.\d+)?%)',
            # Negative numbers
            r'(-\d+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                answer = matches[-1]
                if answer:
                    return self._clean_extracted_answer(answer)

        return None

    def _parse_comparison_symbols(self, text: str, task_type: str = "general") -> Optional[str]:
        """Extract comparison results from symbols."""
        if task_type != "comparison":
            return None

        # Look for comparison expressions
        comparison_patterns = [
            r'(\d+)\s*([<>=])\s*(\d+)',
            r'(greater than|less than|equal to)',
            r'([<>=])',
        ]

        for pattern in comparison_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return self._normalize_comparison_result(matches[-1])

        return None

    def _parse_fallback_numbers(self, text: str, task_type: str = "general") -> Optional[str]:
        """Extract numbers as fallback when other methods fail."""
        # Find all numbers in the text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)

        if numbers:
            # Return the last number found (likely to be the answer)
            return numbers[-1]

        return None

    def _clean_extracted_answer(self, answer: str) -> str:
        """Clean and normalize extracted answer."""
        # Remove common prefixes/suffixes
        answer = re.sub(r'^(?:is\s+|equals?\s+|=\s*)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'(?:\s*[,.]?\s*$)', '', answer)

        # Remove LaTeX commands
        answer = re.sub(r'\\[a-zA-Z]+{?([^}]*)}?', r'\1', answer)

        # Clean whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()

        return answer

    def _process_comparison_match(self, match: tuple, task_type: str) -> Optional[str]:
        """Process comparison matches specifically."""
        if len(match) >= 2:
            # Extract comparison result
            return self._normalize_comparison_result(match[1] if len(match) > 1 else match[0])
        return None

    def _normalize_comparison_result(self, result: Union[str, tuple]) -> str:
        """Normalize comparison results to standard format."""
        if isinstance(result, tuple):
            result = ' '.join(str(x) for x in result)

        result = str(result).lower().strip()

        # Normalize to standard phrases
        if any(word in result for word in ['greater', '>', 'larger', 'bigger']):
            return "greater than"
        elif any(word in result for word in ['less', '<', 'smaller']):
            return "less than"
        elif any(word in result for word in ['equal', '=', 'same']):
            return "equal to"

        return result

    def _convert_to_expected_type(self, value: str, expected_type: str):
        """Convert extracted value to expected type."""
        if expected_type == "str" or not value:
            return value

        # Try to convert to number
        try:
            # Handle fractions
            if '/' in value and expected_type in ["float", "auto"]:
                fraction = Fraction(value)
                return float(fraction)

            # Handle percentages
            if '%' in value:
                return float(value.replace('%', '')) / 100

            # Try integer first
            if expected_type in ["int", "auto"]:
                try:
                    return int(float(value))
                except (ValueError, OverflowError):
                    pass

            # Try float
            if expected_type in ["float", "auto"]:
                return float(value)

        except (ValueError, TypeError, ZeroDivisionError):
            pass

        # Return as string if conversion fails
        return value

    def get_parsing_debug_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about parsing attempts for debugging."""
        return self.parsing_attempts.copy()


class ComparisonParser(AnswerParser):
    """Specialized parser for comparison tasks."""

    def parse_comparison_result(self, response: str) -> Optional[str]:
        """Parse comparison result specifically."""
        return self.parse_answer(response, expected_type="str", task_type="comparison")


class SequenceParser(AnswerParser):
    """Specialized parser for sequence completion tasks."""

    def parse_sequence_answer(self, response: str, sequence_type: str = "general") -> Optional[Union[int, float]]:
        """Parse sequence completion answer."""
        result = self.parse_answer(response, expected_type="auto", task_type=sequence_type)

        # Try to convert to number for sequence tasks
        if isinstance(result, str):
            try:
                if '.' in result:
                    return float(result)
                else:
                    return int(result)
            except (ValueError, TypeError):
                pass

        return result