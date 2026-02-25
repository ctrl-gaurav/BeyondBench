"""
Sequence parsing utilities for beyondbench package.

This module provides specialized parsing for sequence completion tasks,
particularly for medium suite tasks that involve mathematical sequences.
"""

import re
import json
import logging
from typing import Union, Optional, List, Any
from .parsing import parse_boxed_answer, parse_number, parse_number_list
from .shared_utils import is_valid_number, round_if_close_to_int


class SequenceAnswerParser:
    """
    Enhanced parser for sequence completion tasks with robust extraction capabilities.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the sequence parser.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.logger = logging.getLogger(__name__)

    def parse_sequence_answer(
        self,
        response: str,
        sequence_type: str = "general",
        expected_type: str = "auto"
    ) -> Optional[Union[int, float, List]]:
        """
        Parse sequence completion answer from model response.

        Args:
            response: Model response text
            sequence_type: Type of sequence ("fibonacci", "prime", "geometric", etc.)
            expected_type: Expected answer type ("int", "float", "list", "auto")

        Returns:
            Parsed answer or None if parsing fails
        """
        if not response:
            return None

        # Try multiple parsing strategies in order of preference
        result = (
            self._parse_with_unified_parser(response, expected_type) or
            self._parse_sequence_specific(response, sequence_type) or
            self._parse_mathematical_expression(response) or
            self._parse_from_context(response, sequence_type)
        )

        if result is not None and expected_type in ["int", "number"]:
            # Convert to integer if expected
            try:
                if isinstance(result, (int, float)):
                    return round_if_close_to_int(result)
                elif isinstance(result, str) and is_valid_number(result):
                    return round_if_close_to_int(float(result))
            except (ValueError, TypeError):
                pass

        return result

    def _parse_with_unified_parser(self, response: str, expected_type: str) -> Optional[Any]:
        """Use the unified parsing system first."""
        try:
            if expected_type == "list":
                return parse_number_list(response)
            else:
                return parse_boxed_answer(response, expected_type)
        except Exception as e:
            if self.debug:
                self.logger.debug(f"Unified parser failed: {e}")
            return None

    def _parse_sequence_specific(self, response: str, sequence_type: str) -> Optional[Union[int, float]]:
        """Parse using sequence-specific patterns."""
        patterns = {
            "fibonacci": [
                r"(?:next|following|subsequent)\s+(?:term|number|value|element)\s+(?:is|would be|equals?)\s+([+-]?\d+(?:\.\d+)?)",
                r"(?:fibonacci|fib)\s+(?:number|term|sequence)\s+(?:is|equals?)\s+([+-]?\d+(?:\.\d+)?)",
                r"(?:answer|result|solution)\s+(?:is|:)\s+([+-]?\d+(?:\.\d+)?)"
            ],
            "prime": [
                r"(?:next|following)\s+prime\s+(?:number|is)\s+([+-]?\d+)",
                r"(?:prime|answer)\s+(?:is|:)\s+([+-]?\d+)",
                r"(?:sequence\s+continues?\s+with|followed\s+by)\s+([+-]?\d+)"
            ],
            "geometric": [
                r"(?:next|following)\s+term\s+(?:is|would be|equals?)\s+([+-]?\d+(?:\.\d+)?)",
                r"(?:geometric\s+sequence|ratio)\s+.*?([+-]?\d+(?:\.\d+)?)",
                r"(?:answer|result)\s+(?:is|:)\s+([+-]?\d+(?:\.\d+)?)"
            ]
        }

        # Get patterns for this sequence type, fallback to general patterns
        sequence_patterns = patterns.get(sequence_type, patterns["fibonacci"])

        for pattern in sequence_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    return parse_number(matches[-1])  # Return the last match
                except (ValueError, TypeError):
                    continue

        return None

    def _parse_mathematical_expression(self, response: str) -> Optional[Union[int, float]]:
        """Parse mathematical expressions and equations."""
        # Look for explicit mathematical patterns
        math_patterns = [
            r"=\s*([+-]?\d+(?:\.\d+)?)\s*$",  # Equals sign at end of line
            r":\s*([+-]?\d+(?:\.\d+)?)\s*$",  # Colon at end of line
            r"(?:answer|result|solution|value)\s*[=:]\s*([+-]?\d+(?:\.\d+)?)",
            r"(?:therefore|thus|hence|so)\s*[,:]?\s*([+-]?\d+(?:\.\d+)?)",
            r"([+-]?\d+(?:\.\d+)?)\s*(?:is\s+the|would\s+be\s+the)?\s*(?:answer|result|solution|next\s+term)"
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    return parse_number(matches[-1])
                except (ValueError, TypeError):
                    continue

        return None

    def _parse_from_context(self, response: str, sequence_type: str) -> Optional[Union[int, float]]:
        """Parse using contextual clues and final fallback."""
        # Split response into lines and check the last few lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        if not lines:
            return None

        # Check last 3 lines for standalone numbers
        for i in range(min(3, len(lines))):
            line = lines[-(i+1)]

            # Look for standalone numbers
            standalone_number = re.search(r'^([+-]?\d+(?:\.\d+)?)$', line)
            if standalone_number:
                try:
                    return parse_number(standalone_number.group(1))
                except (ValueError, TypeError):
                    continue

            # Look for numbers at the end of sentences
            end_number = re.search(r'([+-]?\d+(?:\.\d+)?)\s*[.!]?\s*$', line)
            if end_number:
                try:
                    return parse_number(end_number.group(1))
                except (ValueError, TypeError):
                    continue

        return None


# Convenience function for backward compatibility
def parse_sequence_answer(response: str, sequence_type: str = "general") -> Optional[Union[int, float]]:
    """
    Parse sequence answer using the SequenceAnswerParser.

    Args:
        response: Model response text
        sequence_type: Type of sequence

    Returns:
        Parsed answer or None
    """
    parser = SequenceAnswerParser(debug=False)
    return parser.parse_sequence_answer(response, sequence_type, "auto")


def parse_fibonacci_answer(response: str) -> Optional[int]:
    """Parse Fibonacci sequence answer."""
    result = parse_sequence_answer(response, "fibonacci")
    if isinstance(result, (int, float)):
        return int(round_if_close_to_int(result))
    return None


def parse_prime_answer(response: str) -> Optional[int]:
    """Parse prime sequence answer."""
    result = parse_sequence_answer(response, "prime")
    if isinstance(result, (int, float)):
        return int(round_if_close_to_int(result))
    return None


def parse_geometric_answer(response: str) -> Optional[Union[int, float]]:
    """Parse geometric sequence answer."""
    return parse_sequence_answer(response, "geometric")


def parse_algebraic_answer(response: str) -> Optional[Union[int, float]]:
    """Parse algebraic sequence answer."""
    return parse_sequence_answer(response, "algebraic")


def validate_sequence_answer(
    parsed_answer: Union[int, float],
    expected_answer: Union[int, float],
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that a parsed sequence answer matches the expected answer.

    Args:
        parsed_answer: The parsed answer
        expected_answer: The expected correct answer
        tolerance: Tolerance for float comparison

    Returns:
        bool: True if answers match within tolerance
    """
    if parsed_answer is None or expected_answer is None:
        return parsed_answer == expected_answer

    try:
        parsed_num = float(parsed_answer)
        expected_num = float(expected_answer)
        return abs(parsed_num - expected_num) <= tolerance
    except (ValueError, TypeError):
        return str(parsed_answer) == str(expected_answer)