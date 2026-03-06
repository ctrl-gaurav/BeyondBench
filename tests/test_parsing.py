"""
Test parsing utilities.
"""

import pytest
from beyondbench.utils.parsing import (
    parse_boxed_answer,
    extract_number,
    extract_list,
    parse_number,
    parse_number_list,
    parse_boolean,
    parse_grid,
    robust_extract_answer,
    extract_all_numbers,
    is_valid_number,
    validate_arithmetic_result,
)


class TestParseBoxedAnswer:
    """Test parse_boxed_answer function."""

    def test_simple_boxed_number(self):
        """Test parsing simple boxed number."""
        result = parse_boxed_answer(r"The answer is \boxed{42}", "number")
        assert result == 42

    def test_boxed_with_text(self):
        """Test parsing boxed answer with surrounding text."""
        text = r"After calculation, we get \boxed{123}. This is the final answer."
        result = parse_boxed_answer(text, "number")
        assert result == 123

    def test_boxed_negative_number(self):
        """Test parsing negative number in box."""
        result = parse_boxed_answer(r"\boxed{-15}", "number")
        assert result == -15

    def test_boxed_float(self):
        """Test parsing floating point number."""
        result = parse_boxed_answer(r"\boxed{3.14}", "number")
        assert result == 3.14

    def test_boxed_list(self):
        """Test parsing list in box."""
        result = parse_boxed_answer(r"\boxed{[1, 2, 3]}", "list")
        assert result == [1, 2, 3]

    def test_no_box_fallback(self):
        """Test fallback when no box is present."""
        result = parse_boxed_answer("The answer is 42", "number")
        # Should still try to extract the number
        assert result == 42

    def test_empty_response(self):
        """Test handling empty response."""
        result = parse_boxed_answer("", "number")
        assert result is None

    def test_dollar_wrapped_boxed(self):
        """Test parsing dollar-wrapped boxed."""
        result = parse_boxed_answer(r"$\boxed{99}$", "number")
        assert result == 99

    def test_boxed_with_commas(self):
        """Test parsing number with thousand separators."""
        result = parse_boxed_answer(r"\boxed{1,000,000}", "number")
        assert result == 1000000

    def test_nested_boxed(self):
        """Test parsing nested boxed."""
        result = parse_boxed_answer(r"\boxed{\boxed{55}}", "number")
        assert result == 55

    def test_answer_tags(self):
        """Test parsing answer tags."""
        result = parse_boxed_answer("<answer>42</answer>", "number")
        assert result == 42


class TestExtractNumber:
    """Test extract_number function."""

    def test_extract_integer(self):
        """Test extracting integer from text."""
        result = extract_number("The result is 42 units")
        assert result == 42

    def test_extract_negative(self):
        """Test extracting negative number."""
        result = extract_number("Temperature dropped to -15 degrees")
        assert result == -15

    def test_extract_float(self):
        """Test extracting float."""
        result = extract_number("Pi is approximately 3.14159")
        assert abs(result - 3.14159) < 0.0001

    def test_extract_first_number(self):
        """Test that first number is extracted when multiple present."""
        result = extract_number("We have 5 apples and 3 oranges")
        assert result == 5

    def test_no_number_returns_none(self):
        """Test that None is returned when no number present."""
        result = extract_number("No numbers here")
        assert result is None

    def test_boxed_number(self):
        """Test extracting number from boxed format."""
        result = extract_number(r"\boxed{123}")
        assert result == 123

    def test_thousand_separators(self):
        """Test extracting number with thousand separators."""
        result = extract_number("The population is 1,234,567")
        assert result == 1234567


class TestExtractList:
    """Test extract_list function."""

    def test_extract_bracket_list(self):
        """Test extracting list with brackets."""
        result = extract_list("[1, 2, 3, 4, 5]")
        assert result == [1, 2, 3, 4, 5]

    def test_extract_comma_separated(self):
        """Test extracting comma-separated values."""
        result = extract_list("1, 2, 3, 4, 5")
        assert result == [1, 2, 3, 4, 5]

    def test_extract_space_separated(self):
        """Test extracting space-separated values."""
        result = extract_list("1 2 3 4 5")
        assert result == [1, 2, 3, 4, 5]

    def test_empty_list(self):
        """Test handling empty list."""
        result = extract_list("[]")
        # Empty list should return None or empty list
        assert result is None or result == []

    def test_tuple_notation(self):
        """Test parsing tuple notation."""
        result = extract_list("(1, 2, 3)")
        assert result == [1, 2, 3]

    def test_boxed_list(self):
        """Test extracting list from boxed format."""
        result = extract_list(r"\boxed{[10, 20, 30]}")
        assert result == [10, 20, 30]


class TestParseNumber:
    """Test parse_number function."""

    def test_integer(self):
        """Test parsing integer."""
        assert parse_number("42") == 42

    def test_negative_integer(self):
        """Test parsing negative integer."""
        assert parse_number("-17") == -17

    def test_float(self):
        """Test parsing float."""
        assert parse_number("3.14") == 3.14

    def test_scientific_notation(self):
        """Test parsing scientific notation."""
        assert parse_number("1e10") == 10000000000

    def test_fraction(self):
        """Test parsing fraction."""
        assert parse_number("1/2") == 0.5

    def test_thousand_separator(self):
        """Test parsing with thousand separator."""
        assert parse_number("1,000,000") == 1000000

    def test_with_latex(self):
        """Test parsing with LaTeX artifacts."""
        assert parse_number(r"\textbf{42}") == 42

    def test_none_input(self):
        """Test handling None input."""
        assert parse_number(None) is None

    def test_empty_string(self):
        """Test handling empty string."""
        assert parse_number("") is None


class TestParseNumberList:
    """Test parse_number_list function."""

    def test_json_array(self):
        """Test parsing JSON array."""
        assert parse_number_list("[1, 2, 3]") == [1, 2, 3]

    def test_comma_separated(self):
        """Test parsing comma-separated."""
        assert parse_number_list("1, 2, 3") == [1, 2, 3]

    def test_space_separated(self):
        """Test parsing space-separated."""
        assert parse_number_list("1 2 3") == [1, 2, 3]

    def test_with_negatives(self):
        """Test parsing with negative numbers."""
        assert parse_number_list("[-1, 0, 1]") == [-1, 0, 1]

    def test_mixed_separators(self):
        """Test parsing with mixed separators."""
        result = parse_number_list("1, 2; 3  4")
        assert result == [1, 2, 3, 4]

    def test_already_list(self):
        """Test parsing when input is already a list."""
        assert parse_number_list([1, 2, 3]) == [1, 2, 3]


class TestParseBoolean:
    """Test parse_boolean function."""

    def test_true_values(self):
        """Test various true values."""
        assert parse_boolean("true") is True
        assert parse_boolean("True") is True
        assert parse_boolean("yes") is True
        assert parse_boolean("1") is True
        assert parse_boolean("correct") is True

    def test_false_values(self):
        """Test various false values."""
        assert parse_boolean("false") is False
        assert parse_boolean("False") is False
        assert parse_boolean("no") is False
        assert parse_boolean("0") is False
        assert parse_boolean("incorrect") is False

    def test_invalid_values(self):
        """Test invalid boolean values."""
        assert parse_boolean("maybe") is None
        assert parse_boolean("") is None


class TestParseGrid:
    """Test parse_grid function."""

    def test_json_grid(self):
        """Test parsing JSON grid."""
        result = parse_grid("[[1, 2], [3, 4]]")
        assert result == [[1, 2], [3, 4]]

    def test_line_separated(self):
        """Test parsing line-separated grid."""
        result = parse_grid("1 2 3\n4 5 6\n7 8 9")
        assert result == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_pipe_separated(self):
        """Test parsing pipe-separated grid."""
        result = parse_grid("1|2|3\n4|5|6")
        assert result == [[1, 2, 3], [4, 5, 6]]


class TestRobustExtractAnswer:
    """Test robust_extract_answer function."""

    def test_boxed_instruction_followed(self):
        """Test that boxed format is recognized as instruction followed."""
        followed, result = robust_extract_answer(r"\boxed{42}", "number")
        assert followed is True
        assert result == 42

    def test_non_boxed_instruction_not_followed(self):
        """Test that non-boxed format is recognized as instruction not followed."""
        followed, result = robust_extract_answer("The answer is 42", "number")
        assert followed is False
        assert result == 42

    def test_auto_detect_number(self):
        """Test auto-detection of number type."""
        followed, result = robust_extract_answer(r"\boxed{123}", "auto")
        assert result == 123

    def test_auto_detect_list(self):
        """Test auto-detection of list type."""
        followed, result = robust_extract_answer(r"\boxed{[1, 2, 3]}", "auto")
        assert result == [1, 2, 3]


class TestExtractAllNumbers:
    """Test extract_all_numbers function."""

    def test_multiple_numbers(self):
        """Test extracting multiple numbers."""
        result = extract_all_numbers("There are 5 apples, 3 oranges, and 7 bananas")
        assert result == [5, 3, 7]

    def test_mixed_integers_floats(self):
        """Test extracting mixed integers and floats."""
        result = extract_all_numbers("Values: 1, 2.5, 3, 4.75")
        assert result == [1, 2.5, 3, 4.75]

    def test_with_thousand_separators(self):
        """Test extracting numbers with thousand separators."""
        result = extract_all_numbers("Prices: $1,000 and $2,500,000")
        assert result == [1000, 2500000]


class TestIsValidNumber:
    """Test is_valid_number function."""

    def test_integer(self):
        """Test integer is valid."""
        assert is_valid_number(42) is True

    def test_float(self):
        """Test float is valid."""
        assert is_valid_number(3.14) is True

    def test_none(self):
        """Test None is not valid."""
        assert is_valid_number(None) is False

    def test_boolean(self):
        """Test boolean is not valid (even though True/False are int-like)."""
        assert is_valid_number(True) is False

    def test_string_number(self):
        """Test string number is valid."""
        assert is_valid_number("42") is True


class TestValidateArithmeticResult:
    """Test validate_arithmetic_result function."""

    def test_exact_match(self):
        """Test exact match."""
        assert validate_arithmetic_result(42, 42) is True

    def test_within_tolerance(self):
        """Test within tolerance."""
        assert validate_arithmetic_result(3.14159, 3.14159265, tolerance=0.001) is True

    def test_outside_tolerance(self):
        """Test outside tolerance."""
        assert validate_arithmetic_result(3.0, 4.0, tolerance=0.001) is False


class TestEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_markdown_bold_number(self):
        """Test extracting number from markdown bold."""
        result = parse_boxed_answer("The final answer is **42**", "number")
        assert result == 42

    def test_multiple_boxed_takes_last(self):
        """Test that multiple boxed answers take the last one."""
        result = parse_boxed_answer(r"First \boxed{1}, then \boxed{2}, finally \boxed{3}", "number")
        assert result == 3

    def test_code_block_extraction(self):
        """Test extracting from code block."""
        text = """
        ```python
        answer = 42
        ```
        """
        result = parse_boxed_answer(text, "number")
        assert result == 42

    def test_latex_math_mode(self):
        """Test extracting from LaTeX math mode."""
        result = parse_boxed_answer(r"The solution is $42$", "number")
        assert result == 42

    def test_very_long_response(self):
        """Test handling very long response."""
        long_text = "Step " * 1000 + r"\boxed{999}"
        result = parse_boxed_answer(long_text, "number")
        assert result == 999

    def test_unicode_numbers(self):
        """Test handling unicode that looks like numbers."""
        # This should not be parsed as a number
        result = parse_number("①②③")
        # May return None or the first digit equivalent
        # The important thing is it doesn't crash

    def test_negative_list(self):
        """Test list with negative numbers."""
        result = extract_list("[-5, -3, -1, 0, 1, 3, 5]")
        assert result == [-5, -3, -1, 0, 1, 3, 5]

    def test_float_list(self):
        """Test list with float numbers."""
        result = extract_list("[1.1, 2.2, 3.3]")
        assert result == [1.1, 2.2, 3.3]


class TestComparisonParsing:
    """Test comparison-specific parsing."""

    def test_greater_than_boxed(self):
        result = parse_boxed_answer(r"\boxed{greater than}", "string")
        assert result is not None
        assert "greater" in str(result).lower()

    def test_less_than_natural_language(self):
        from beyondbench.utils.parsing import extract_from_final_answer
        result = extract_from_final_answer("After comparing, the answer is less than.")
        assert result is not None


class TestSequenceParsing:
    """Test sequence answer parsing."""

    def test_fibonacci_answer(self):
        from beyondbench.utils.sequence_parsing_utils import parse_fibonacci_answer
        result = parse_fibonacci_answer(r"The next Fibonacci number is \boxed{21}")
        assert result == 21

    def test_sequence_parser_class(self):
        from beyondbench.utils.sequence_parsing_utils import SequenceAnswerParser
        parser = SequenceAnswerParser()
        result = parser.parse_sequence_answer(r"\boxed{55}", "fibonacci")
        assert result == 55


class TestGridParsing:
    """Test grid/matrix parsing."""

    def test_sudoku_grid(self):
        grid_text = "1 2 3\n4 5 6\n7 8 9"
        result = parse_grid(grid_text)
        assert result == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_json_grid(self):
        result = parse_grid("[[1,2],[3,4]]")
        assert result == [[1, 2], [3, 4]]


class TestRobustExtraction:
    """Test robust_extract_answer edge cases."""

    def test_markdown_bold_answer(self):
        result = parse_boxed_answer("The final answer is **42**", "number")
        assert result == 42

    def test_code_block_answer(self):
        result = parse_boxed_answer("```\nanswer = 42\n```", "number")
        assert result == 42

    def test_answer_tag(self):
        result = parse_boxed_answer("<answer>42</answer>", "number")
        assert result == 42

    def test_latex_text_in_boxed(self):
        result = parse_boxed_answer(r"\boxed{\text{42}}", "number")
        assert result == 42
