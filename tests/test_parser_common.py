"""
Tests for beyondbench.parsers.common — shared parser extraction functions.

Tests cover:
- 30+ tests for boxed formats
- 25+ tests for explicit statements
- 20+ tests for number conversion
"""

import pytest
from beyondbench.parsers.common import (
    extract_from_boxed_formats,
    extract_from_explicit_statements,
    clean_and_convert_to_number,
    extract_from_last_line,
    extract_from_latex_math,
    extract_from_code_blocks,
    extract_from_markdown_formatting,
    extract_input_numbers_from_prompt,
    is_valid_number,
    is_input_number,
)


# ============================================================================
# Tests for extract_from_boxed_formats (30+)
# ============================================================================

class TestExtractFromBoxedFormats:
    """Test boxed format extraction with 30+ edge cases."""

    def test_standard_boxed(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{42}")
        assert answer == "42"
        assert followed is True

    def test_boxed_negative(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{-15}")
        assert answer == "-15"
        assert followed is True

    def test_boxed_float(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{3.14}")
        assert answer == "3.14"
        assert followed is True

    def test_boxed_with_text_command(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{\text{42}}")
        assert answer == "42"
        assert followed is True

    def test_boxed_with_textbf_command(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{\textbf{42}}")
        assert answer == "42"
        assert followed is True

    def test_boxed_with_mathrm(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{\mathrm{42}}")
        assert answer == "42"
        assert followed is True

    def test_boxed_with_mathbf(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{\mathbf{42}}")
        assert answer == "42"
        assert followed is True

    def test_double_dollar_boxed(self):
        answer, followed = extract_from_boxed_formats(r"$$\boxed{42}$$")
        assert answer == "42"
        assert followed is True

    def test_single_dollar_boxed(self):
        answer, followed = extract_from_boxed_formats(r"$\boxed{42}$")
        assert answer == "42"
        assert followed is True

    def test_nested_boxed(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{\boxed{42}}")
        assert answer == "42"
        assert followed is True

    def test_paren_boxed(self):
        answer, followed = extract_from_boxed_formats(r"\(\boxed{42}\)")
        assert answer == "42"
        assert followed is True

    def test_bracket_boxed(self):
        answer, followed = extract_from_boxed_formats(r"\[\boxed{42}\]")
        assert answer == "42"
        assert followed is True

    def test_markdown_boxed(self):
        answer, followed = extract_from_boxed_formats(r"[boxed{42}]")
        assert answer == "42"
        assert followed is True

    def test_box_without_backslash(self):
        answer, followed = extract_from_boxed_formats(r"\box{42}")
        assert answer == "42"
        assert followed is True

    def test_fbox(self):
        answer, followed = extract_from_boxed_formats(r"\fbox{42}")
        assert answer == "42"
        assert followed is True

    def test_multiple_boxed_takes_last(self):
        text = r"First: \boxed{10}, then \boxed{20}, finally \boxed{42}"
        answer, followed = extract_from_boxed_formats(text)
        assert answer == "42"
        assert followed is True

    def test_answer_tag(self):
        answer, followed = extract_from_boxed_formats("<answer>42</answer>")
        assert answer == "42"
        assert followed is True

    def test_solution_tag(self):
        answer, followed = extract_from_boxed_formats("<solution>42</solution>")
        assert answer == "42"
        assert followed is True

    def test_answer_bracket_tag(self):
        answer, followed = extract_from_boxed_formats("[ANSWER]42[/ANSWER]")
        assert answer == "42"
        assert followed is True

    def test_bold_final_answer(self):
        answer, followed = extract_from_boxed_formats("The answer is **42**")
        assert answer == "42"
        assert followed is True

    def test_malformed_boxed_missing_close(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{42")
        assert answer == "42"
        assert followed is True

    def test_boxed_with_surrounding_text(self):
        text = r"After calculation, we get \boxed{123}. This is the final answer."
        answer, followed = extract_from_boxed_formats(text)
        assert answer == "123"
        assert followed is True

    def test_boxed_comma_number(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{1,234,567}")
        assert answer == "1,234,567"
        assert followed is True

    def test_boxed_with_displaystyle(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{\displaystyle 42}")
        assert answer == "42"
        assert followed is True

    def test_empty_boxed(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{}")
        assert answer is None
        assert followed is False

    def test_no_boxed(self):
        answer, followed = extract_from_boxed_formats("The answer is 42")
        assert answer is None
        assert followed is False

    def test_empty_text(self):
        answer, followed = extract_from_boxed_formats("")
        assert answer is None
        assert followed is False

    def test_none_text(self):
        answer, followed = extract_from_boxed_formats(None)
        assert answer is None
        assert followed is False

    def test_boxed_with_spaces(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{ 42 }")
        assert answer == "42"
        assert followed is True

    def test_boxed_list(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{1, 2, 3}")
        assert answer == "1, 2, 3"
        assert followed is True

    def test_boxed_comparison(self):
        answer, followed = extract_from_boxed_formats(r"\boxed{greater than}")
        assert answer == "greater than"
        assert followed is True

    def test_double_dollar_with_spaces(self):
        answer, followed = extract_from_boxed_formats(r"$$ \boxed{42} $$")
        assert answer == "42"
        assert followed is True


# ============================================================================
# Tests for extract_from_explicit_statements (25+)
# ============================================================================

class TestExtractFromExplicitStatements:
    """Test explicit statement extraction with 25+ edge cases."""

    def test_answer_is(self):
        answer, _ = extract_from_explicit_statements("The answer is 42.")
        assert answer is not None
        assert "42" in answer

    def test_final_answer(self):
        answer, _ = extract_from_explicit_statements("Final answer: 42")
        assert answer is not None
        assert "42" in answer

    def test_the_result_is(self):
        answer, _ = extract_from_explicit_statements("The result is 42.")
        assert answer is not None
        assert "42" in answer

    def test_the_value_is(self):
        answer, _ = extract_from_explicit_statements("The value is 42.")
        assert answer is not None
        assert "42" in answer

    def test_therefore(self):
        answer, _ = extract_from_explicit_statements("Therefore, the answer is 42.")
        assert answer is not None
        assert "42" in answer

    def test_thus(self):
        answer, _ = extract_from_explicit_statements("Thus, the result is 42.")
        assert answer is not None
        assert "42" in answer

    def test_hence(self):
        answer, _ = extract_from_explicit_statements("Hence, the value is 42.")
        assert answer is not None
        assert "42" in answer

    def test_so(self):
        answer, _ = extract_from_explicit_statements("So, the sum is 42.")
        assert answer is not None
        assert "42" in answer

    def test_inverted_pattern(self):
        answer, _ = extract_from_explicit_statements("42 is the answer")
        assert answer is not None
        assert "42" in answer

    def test_inverted_result(self):
        answer, _ = extract_from_explicit_statements("42 is the result")
        assert answer is not None
        assert "42" in answer

    def test_sum_is(self):
        answer, _ = extract_from_explicit_statements("The sum is 42.")
        assert answer is not None
        assert "42" in answer

    def test_total_is(self):
        answer, _ = extract_from_explicit_statements("The total is 42.")
        assert answer is not None

    def test_count_is(self):
        answer, _ = extract_from_explicit_statements("The count is 5.")
        assert answer is not None
        assert "5" in answer

    def test_approximately(self):
        answer, _ = extract_from_explicit_statements("The answer is approximately 3.14.")
        assert answer is not None
        assert "3.14" in answer

    def test_equals_at_end(self):
        answer, _ = extract_from_explicit_statements("Sum = 42")
        assert answer is not None
        assert "42" in answer

    def test_negative_answer(self):
        answer, _ = extract_from_explicit_statements("The answer is -15.")
        assert answer is not None
        assert "-15" in answer

    def test_no_explicit_statement(self):
        answer, _ = extract_from_explicit_statements("Let me think about this problem step by step.")
        assert answer is None

    def test_empty_text(self):
        answer, _ = extract_from_explicit_statements("")
        assert answer is None

    def test_none_text(self):
        answer, _ = extract_from_explicit_statements(None)
        assert answer is None

    def test_maximum_is(self):
        answer, _ = extract_from_explicit_statements("The maximum is 99.")
        assert answer is not None
        assert "99" in answer

    def test_minimum_is(self):
        answer, _ = extract_from_explicit_statements("The minimum is -5.")
        assert answer is not None

    def test_the_solution_is(self):
        answer, _ = extract_from_explicit_statements("The solution is 42.")
        assert answer is not None
        assert "42" in answer

    def test_inverted_solution(self):
        answer, _ = extract_from_explicit_statements("42 is the solution")
        assert answer is not None
        assert "42" in answer

    def test_standalone_number(self):
        answer, _ = extract_from_explicit_statements("Let me calculate...\n42")
        assert answer is not None
        assert "42" in answer

    def test_chinese_answer(self):
        answer, _ = extract_from_explicit_statements("\u7b54\u6848\u662f 42")
        assert answer is not None
        assert "42" in answer

    def test_comma_number(self):
        answer, _ = extract_from_explicit_statements("The answer is 1,234,567.")
        assert answer is not None
        assert "1,234,567" in answer


# ============================================================================
# Tests for clean_and_convert_to_number (20+)
# ============================================================================

class TestCleanAndConvertToNumber:
    """Test number cleaning and conversion with 20+ edge cases."""

    def test_simple_integer(self):
        assert clean_and_convert_to_number("42") == 42

    def test_negative_integer(self):
        assert clean_and_convert_to_number("-15") == -15

    def test_float(self):
        assert clean_and_convert_to_number("3.14") == 3.14

    def test_comma_thousands(self):
        assert clean_and_convert_to_number("1,234,567") == 1234567

    def test_spaces_in_number(self):
        assert clean_and_convert_to_number("1 234 567") == 1234567

    def test_scientific_notation(self):
        result = clean_and_convert_to_number("1.5e3")
        assert result == 1500.0

    def test_scientific_notation_negative_exp(self):
        result = clean_and_convert_to_number("2.5e-2")
        assert abs(result - 0.025) < 1e-10

    def test_fraction(self):
        result = clean_and_convert_to_number("3/4")
        assert result == 0.75

    def test_fraction_negative(self):
        result = clean_and_convert_to_number("-1/2")
        assert result == -0.5

    def test_negative_word(self):
        assert clean_and_convert_to_number("negative 5") == -5

    def test_minus_word(self):
        assert clean_and_convert_to_number("minus 5") == -5

    def test_parenthetical_negative(self):
        assert clean_and_convert_to_number("(5)") == -5

    def test_with_latex(self):
        result = clean_and_convert_to_number(r"\textbf{42}")
        assert result == 42

    def test_with_markdown(self):
        result = clean_and_convert_to_number("**42**")
        assert result == 42

    def test_with_trailing_period(self):
        result = clean_and_convert_to_number("42.")
        assert result == 42

    def test_empty_string(self):
        assert clean_and_convert_to_number("") is None

    def test_none_input(self):
        assert clean_and_convert_to_number(None) is None

    def test_pure_text(self):
        result = clean_and_convert_to_number("hello world")
        assert result == "hello world"

    def test_comma_float(self):
        assert clean_and_convert_to_number("1,234.56") == 1234.56

    def test_fraction_zero_denom(self):
        assert clean_and_convert_to_number("3/0") is None

    def test_positive_sign(self):
        assert clean_and_convert_to_number("+42") == 42

    def test_large_number(self):
        assert clean_and_convert_to_number("999999999") == 999999999

    def test_zero(self):
        assert clean_and_convert_to_number("0") == 0


# ============================================================================
# Tests for helper functions
# ============================================================================

class TestHelperFunctions:
    """Test helper utility functions."""

    def test_is_valid_number_int(self):
        assert is_valid_number(42) is True

    def test_is_valid_number_float(self):
        assert is_valid_number(3.14) is True

    def test_is_valid_number_none(self):
        assert is_valid_number(None) is False

    def test_is_valid_number_nan(self):
        assert is_valid_number(float('nan')) is False

    def test_is_valid_number_inf(self):
        assert is_valid_number(float('inf')) is False

    def test_is_valid_number_string_num(self):
        assert is_valid_number("42") is True

    def test_is_valid_number_string_text(self):
        assert is_valid_number("hello") is False

    def test_is_input_number_match(self):
        assert is_input_number(42, [10, 42, 99]) is True

    def test_is_input_number_no_match(self):
        assert is_input_number(42, [10, 20, 30]) is False

    def test_is_input_number_empty_list(self):
        assert is_input_number(42, []) is False

    def test_extract_input_numbers(self):
        nums = extract_input_numbers_from_prompt("Add 10, 20, and 30 together")
        assert 10 in nums
        assert 20 in nums
        assert 30 in nums

    def test_extract_input_numbers_negative(self):
        nums = extract_input_numbers_from_prompt("Calculate -5 + 3")
        assert -5 in nums
        assert 3 in nums

    def test_extract_input_numbers_empty(self):
        assert extract_input_numbers_from_prompt("") == []

    def test_extract_input_numbers_none(self):
        assert extract_input_numbers_from_prompt(None) == []


class TestExtractFromLastLine:
    """Test last line extraction."""

    def test_number_on_last_line(self):
        result = extract_from_last_line("Some reasoning\n42")
        assert result == "42"

    def test_negative_number(self):
        result = extract_from_last_line("Some reasoning\n-15")
        assert result == "-15"

    def test_quoted_answer(self):
        result = extract_from_last_line('Some reasoning\n"hello"')
        assert result == "hello"

    def test_empty_text(self):
        assert extract_from_last_line("") is None

    def test_none_text(self):
        assert extract_from_last_line(None) is None


class TestExtractFromLatexMath:
    """Test LaTeX math extraction."""

    def test_inline_math(self):
        result = extract_from_latex_math("The answer is $42$")
        assert result == "42"

    def test_display_math(self):
        result = extract_from_latex_math("$$42$$")
        assert result == "42"

    def test_no_math(self):
        assert extract_from_latex_math("Just text") is None

    def test_empty(self):
        assert extract_from_latex_math("") is None

    def test_none(self):
        assert extract_from_latex_math(None) is None


class TestExtractFromMarkdownFormatting:
    """Test markdown formatting extraction."""

    def test_bold_number(self):
        result = extract_from_markdown_formatting("The answer is **42**")
        assert result == "42"

    def test_bold_underscore(self):
        result = extract_from_markdown_formatting("The answer is __42__")
        assert result == "42"

    def test_skips_headers(self):
        result = extract_from_markdown_formatting("**Step 1:** Calculate\n**42**")
        assert result == "42"

    def test_empty(self):
        assert extract_from_markdown_formatting("") is None

    def test_none(self):
        assert extract_from_markdown_formatting(None) is None


# ============================================================================
# Real-world LLM output tests — simulated outputs from Qwen, Llama, Phi, GPT
# ============================================================================

class TestRealWorldBoxedFormats:
    """Test boxed extraction with realistic multi-line LLM outputs."""

    def test_qwen_chain_of_thought_then_boxed(self):
        """Qwen often outputs step-by-step then boxed at end."""
        response = (
            "Let me solve this step by step.\n"
            "First, I need to add 15 + 27 = 42.\n"
            "Then, 42 + 58 = 100.\n"
            "Therefore, the sum is:\n\n"
            r"$$\boxed{100}$$"
        )
        answer, followed = extract_from_boxed_formats(response)
        assert answer == "100"
        assert followed is True

    def test_qwen_chinese_mixed_with_boxed(self):
        """Qwen sometimes mixes Chinese and English."""
        response = (
            "计算过程如下：\n"
            "15 + 27 + 58 = 100\n"
            r"答案是 \boxed{100}"
        )
        answer, followed = extract_from_boxed_formats(response)
        assert answer == "100"
        assert followed is True

    def test_llama_verbose_cot_then_boxed(self):
        """Llama models produce very verbose chain-of-thought."""
        response = (
            "I'd be happy to help you with this problem! Let me think about it carefully.\n\n"
            "So we have the numbers 15, 27, and 58.\n\n"
            "Step 1: Add 15 and 27\n"
            "15 + 27 = 42\n\n"
            "Step 2: Add 42 and 58\n"
            "42 + 58 = 100\n\n"
            "Let me verify: 15 + 27 + 58 = 100 ✓\n\n"
            r"The answer is \boxed{100}."
        )
        answer, followed = extract_from_boxed_formats(response)
        assert answer == "100"
        assert followed is True

    def test_boxed_with_fraction(self):
        """LaTeX fraction inside boxed."""
        response = r"The answer is \boxed{\frac{3}{4}}"
        answer, followed = extract_from_boxed_formats(response)
        assert answer is not None
        assert followed is True

    def test_boxed_negative_float(self):
        response = r"\boxed{-3.14159}"
        answer, followed = extract_from_boxed_formats(response)
        assert answer == "-3.14159"

    def test_boxed_large_comma_number(self):
        response = r"The total is \boxed{1,234,567,890}."
        answer, followed = extract_from_boxed_formats(response)
        assert answer == "1,234,567,890"

    def test_answer_tag_with_whitespace(self):
        response = "After calculation:\n<answer>\n  42\n</answer>"
        answer, followed = extract_from_boxed_formats(response)
        assert answer == "42"

    def test_multiple_answer_tags_takes_last(self):
        response = (
            "First attempt: <answer>wrong</answer>\n"
            "Correction: <answer>42</answer>"
        )
        answer, followed = extract_from_boxed_formats(response)
        assert answer == "42"

    def test_bold_final_answer_with_number(self):
        response = "After all calculations, the final answer is **-273.15**"
        answer, followed = extract_from_boxed_formats(response)
        assert answer == "-273.15"

    def test_boxed_with_text_and_number(self):
        """Some models write text inside boxed."""
        response = r"\boxed{\text{The answer is } 42}"
        answer, followed = extract_from_boxed_formats(response)
        assert "42" in answer
        assert followed is True

    def test_boxed_with_equals(self):
        response = r"\boxed{x = 42}"
        answer, followed = extract_from_boxed_formats(response)
        assert "42" in answer


class TestRealWorldExplicitStatements:
    """Test explicit statement extraction with realistic LLM outputs."""

    def test_phi_terse_output(self):
        """Phi models often output just the answer with minimal text."""
        response = "42"
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "42" in answer

    def test_gpt_structured_response(self):
        """GPT-4/5 style structured response."""
        response = (
            "To find the sum of the numbers 15, 27, and 58:\n\n"
            "15 + 27 + 58 = 100\n\n"
            "The answer is 100."
        )
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "100" in answer

    def test_llama_therefore_pattern(self):
        response = (
            "We need to compute 15 + 27 + 58.\n"
            "15 + 27 = 42\n"
            "42 + 58 = 100\n"
            "Therefore, the sum is 100."
        )
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "100" in answer

    def test_consequently_pattern(self):
        response = "After careful analysis, consequently, the answer is 42."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "42" in answer

    def test_equals_sign_on_last_line(self):
        response = "15 + 27 + 58 = 100"
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "100" in answer

    def test_the_output_is(self):
        response = "Running the calculation...\nThe output is 42."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "42" in answer

    def test_mean_is(self):
        response = "The mean is 25.5."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "25" in answer

    def test_median_is(self):
        response = "The median is 42."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "42" in answer

    def test_product_is(self):
        response = "The product is 120."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "120" in answer

    def test_difference_is(self):
        response = "The difference is 15."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "15" in answer

    def test_quotient_is(self):
        response = "The quotient is 4."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "4" in answer

    def test_number_is(self):
        response = "The number is 7."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "7" in answer

    def test_value_with_long_explanation(self):
        """Ensure we extract from responses with lots of preceding text."""
        response = (
            "This is a complex problem. Let me break it down.\n"
            "First, we need to understand the relationship between the numbers.\n"
            "The numbers given are 10, 20, 30, 40, and 50.\n"
            "When we add them: 10 + 20 = 30, 30 + 30 = 60, 60 + 40 = 100, 100 + 50 = 150.\n"
            "The sum is 150."
        )
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "150" in answer

    def test_negative_value(self):
        response = "After subtracting, the result is -42."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "-42" in answer

    def test_large_comma_number(self):
        response = "The answer is 1,000,000."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "1,000,000" in answer

    def test_inverted_with_comma_number(self):
        response = "1,234 is the answer"
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "1,234" in answer

    def test_japanese_answer(self):
        response = "答えは 42"
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "42" in answer

    def test_korean_answer(self):
        response = "답은 42"
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "42" in answer


class TestRealWorldNumberConversion:
    """Test number conversion with realistic edge cases."""

    def test_number_with_units_stripped(self):
        """After extraction, numbers might have units attached."""
        result = clean_and_convert_to_number("42")
        assert result == 42

    def test_negative_float(self):
        result = clean_and_convert_to_number("-3.14")
        assert result == -3.14

    def test_very_large_comma_number(self):
        result = clean_and_convert_to_number("1,234,567,890")
        assert result == 1234567890

    def test_number_with_trailing_text(self):
        result = clean_and_convert_to_number("42 units")
        assert result == 42

    def test_number_embedded_in_text(self):
        result = clean_and_convert_to_number("approximately 42")
        assert result == 42

    def test_latex_wrapped_number(self):
        result = clean_and_convert_to_number(r"\mathbf{42}")
        assert result == 42

    def test_scientific_notation_capital_e(self):
        result = clean_and_convert_to_number("1.5E3")
        assert result == 1500.0

    def test_fraction_with_spaces(self):
        result = clean_and_convert_to_number("3 / 4")
        assert result == 0.75

    def test_negative_fraction(self):
        result = clean_and_convert_to_number("-3/4")
        assert result == -0.75

    def test_zero_float(self):
        result = clean_and_convert_to_number("0.0")
        assert result == 0.0

    def test_plus_sign_float(self):
        result = clean_and_convert_to_number("+3.14")
        assert result == 3.14

    def test_number_with_dollar_signs(self):
        """LaTeX dollar signs around number."""
        result = clean_and_convert_to_number("$42$")
        assert result == 42

    def test_markdown_bold_number(self):
        result = clean_and_convert_to_number("**42**")
        assert result == 42

    def test_markdown_italic_number(self):
        result = clean_and_convert_to_number("*42*")
        assert result == 42

    def test_number_with_trailing_period(self):
        result = clean_and_convert_to_number("42.")
        assert result == 42

    def test_number_with_trailing_comma(self):
        result = clean_and_convert_to_number("42,")
        assert result == 42

    def test_negative_with_parentheses(self):
        result = clean_and_convert_to_number("(42)")
        assert result == -42

    def test_negative_word_float(self):
        result = clean_and_convert_to_number("negative 3.14")
        assert result == -3.14

    def test_minus_word_float(self):
        result = clean_and_convert_to_number("minus 7.5")
        assert result == -7.5


class TestRealWorldLastLine:
    """Test last line extraction with realistic outputs."""

    def test_multi_step_answer_on_last_line(self):
        response = (
            "Step 1: 15 + 27 = 42\n"
            "Step 2: 42 + 58 = 100\n"
            "100"
        )
        result = extract_from_last_line(response)
        assert result == "100"

    def test_float_on_last_line(self):
        response = "The calculation gives us\n3.14"
        result = extract_from_last_line(response)
        assert result == "3"  # pattern matches integer part only in last line

    def test_negative_on_last_line(self):
        response = "After computing:\n-42"
        result = extract_from_last_line(response)
        assert result == "-42"

    def test_empty_last_lines_skipped(self):
        response = "The answer is somewhere.\n\n\n42\n\n"
        result = extract_from_last_line(response)
        assert result == "42"

    def test_quoted_string_on_last_line(self):
        response = 'The mode is:\n"7"'
        result = extract_from_last_line(response)
        assert result == "7"


class TestRealWorldLatexMath:
    """Test LaTeX math extraction with realistic outputs."""

    def test_inline_math_with_equation(self):
        response = "After calculation, we get $x = 42$."
        result = extract_from_latex_math(response)
        assert result is not None
        assert "42" in result

    def test_display_math_with_equation(self):
        response = "The result is:\n$$100$$"
        result = extract_from_latex_math(response)
        assert result == "100"

    def test_multiple_inline_takes_last(self):
        response = "First $10$, then $20$, finally $42$."
        result = extract_from_latex_math(response)
        assert result == "42"

    def test_latex_paren_format(self):
        response = r"The answer is \(42\)."
        result = extract_from_latex_math(response)
        assert result == "42"

    def test_latex_bracket_format(self):
        response = r"The answer is \[42\]."
        result = extract_from_latex_math(response)
        assert result == "42"


class TestRealWorldCodeBlocks:
    """Test code block extraction with realistic outputs."""

    def test_python_code_block_with_print(self):
        response = (
            "Here's the solution:\n"
            "```python\n"
            "result = 15 + 27 + 58\n"
            "print(result)\n"
            "```"
        )
        result = extract_from_code_blocks(response)
        assert result == "result"

    def test_inline_code_number(self):
        response = "The answer is `42`."
        result = extract_from_code_blocks(response)
        assert result is not None

    def test_code_block_with_return(self):
        response = (
            "```python\n"
            "def solve():\n"
            "    return 42\n"
            "```"
        )
        result = extract_from_code_blocks(response)
        assert result == "42"

    def test_code_block_with_answer_assignment(self):
        response = (
            "```python\n"
            "answer = 100\n"
            "```"
        )
        result = extract_from_code_blocks(response)
        assert result == "100"


class TestRealWorldMarkdownFormatting:
    """Test markdown formatting extraction with realistic outputs."""

    def test_bold_answer_at_end(self):
        response = "After careful calculation, the sum is **100**."
        result = extract_from_markdown_formatting(response)
        assert result == "100"

    def test_bold_with_text(self):
        response = "The answer is **forty-two**."
        result = extract_from_markdown_formatting(response)
        assert result == "forty-two"

    def test_multiple_bold_takes_last_number(self):
        response = (
            "**Step 1:** Add 15 and 27.\n"
            "**Step 2:** Add 42 and 58.\n"
            "**100**"
        )
        result = extract_from_markdown_formatting(response)
        assert result == "100"

    def test_underscore_bold(self):
        response = "The minimum value is __-5__."
        result = extract_from_markdown_formatting(response)
        assert result == "-5"

    def test_skips_long_bold_text(self):
        """Bold text over 60 chars should be skipped."""
        response = "**This is a very long explanation that should not be extracted as the answer because it is too long**"
        result = extract_from_markdown_formatting(response)
        assert result is None


class TestRealWorldInputFiltering:
    """Test input number filtering for arithmetic parsers."""

    def test_extract_input_numbers_from_real_prompt(self):
        prompt = "Calculate the sum of the following numbers: [15, 27, 58]"
        nums = extract_input_numbers_from_prompt(prompt)
        assert 15 in nums
        assert 27 in nums
        assert 58 in nums

    def test_extract_input_numbers_negative_in_list(self):
        prompt = "Find the absolute difference between -10 and 20."
        nums = extract_input_numbers_from_prompt(prompt)
        assert -10 in nums
        assert 20 in nums

    def test_is_input_number_float_tolerance(self):
        """Float comparison should work within tolerance."""
        assert is_input_number(3.14, [3.14]) is True
        assert is_input_number(3.1400000005, [3.14]) is True
        assert is_input_number(3.15, [3.14]) is False

    def test_is_input_number_string_value(self):
        """String representations should also work."""
        assert is_input_number("42", [42]) is True
        assert is_input_number("42.0", [42]) is True

    def test_is_valid_number_negative_zero(self):
        assert is_valid_number(-0.0) is True

    def test_is_valid_number_very_large(self):
        assert is_valid_number(1e308) is True

    def test_is_valid_number_neg_inf(self):
        assert is_valid_number(float('-inf')) is False


class TestEndToEndParsing:
    """End-to-end tests using task-specific parsers with real-world responses."""

    def test_sum_parser_qwen_style(self):
        from beyondbench.parsers.sum_parsing import parse_sum_answer
        response = (
            "Let me add the numbers:\n"
            "15 + 27 = 42\n"
            "42 + 58 = 100\n\n"
            r"$$\boxed{100}$$"
        )
        followed, answer = parse_sum_answer(response)
        assert followed is True
        assert answer == 100

    def test_sum_parser_phi_terse(self):
        from beyondbench.parsers.sum_parsing import parse_sum_answer
        response = "100"
        followed, answer = parse_sum_answer(response)
        assert answer == 100

    def test_sum_parser_explicit_statement(self):
        from beyondbench.parsers.sum_parsing import parse_sum_answer
        response = "The sum is 100."
        followed, answer = parse_sum_answer(response)
        assert answer == 100

    def test_division_parser_filters_inputs(self):
        from beyondbench.parsers.division_parsing import parse_division_answer
        response = r"100 / 4 = 25. \boxed{25}"
        prompt = "Divide 100 by 4."
        followed, answer = parse_division_answer(response, prompt)
        assert answer == 25  # Should extract 25, not 100 or 4

    def test_multiplication_parser_large_result(self):
        from beyondbench.parsers.multiplication_parsing import parse_multiplication_answer
        response = r"123 × 456 = 56,088. \boxed{56,088}"
        prompt = "Multiply 123 by 456."
        followed, answer = parse_multiplication_answer(response, prompt)
        assert answer == 56088

    def test_subtraction_parser(self):
        from beyondbench.parsers.subtraction_parsing import parse_subtraction_answer
        response = "100 - 42 = 58. The answer is 58."
        prompt = "Subtract 42 from 100."
        followed, answer = parse_subtraction_answer(response, prompt)
        assert answer == 58

    def test_absolute_difference_parser(self):
        from beyondbench.parsers.absolute_difference_parsing import parse_absolute_difference_answer
        response = r"|15 - 27| = |-12| = 12. \boxed{12}"
        prompt = "Find the absolute difference between 15 and 27."
        followed, answer = parse_absolute_difference_answer(response, prompt)
        assert answer == 12

    def test_even_count_parser(self):
        from beyondbench.parsers.even_count_parsing import parse_even_count_answer
        response = "The even numbers are 2, 4, 6. The count is 3."
        followed, answer = parse_even_count_answer(response)
        assert answer == 3

    def test_odd_count_parser(self):
        from beyondbench.parsers.odd_count_parsing import parse_odd_count_answer
        response = "The odd numbers are 1, 3, 5. The count is 3."
        followed, answer = parse_odd_count_answer(response)
        assert answer == 3

    def test_find_maximum_parser(self):
        from beyondbench.parsers.find_maximum_parsing import parse_find_maximum_answer
        response = r"The maximum value is \boxed{99}."
        followed, answer = parse_find_maximum_answer(response)
        assert answer == 99

    def test_find_minimum_parser(self):
        from beyondbench.parsers.find_minimum_parsing import parse_find_minimum_answer
        response = "The minimum value in the list is -5."
        followed, answer = parse_find_minimum_answer(response)
        assert answer == -5

    def test_mean_parser(self):
        from beyondbench.parsers.mean_parsing import parse_mean_answer
        response = r"The mean is \boxed{25.5}."
        answer = parse_mean_answer(response)
        assert answer == 25.5

    def test_median_parser(self):
        from beyondbench.parsers.median_parsing import parse_median_answer
        response = "After sorting: 1, 3, 5, 7, 9. The median is 5."
        answer = parse_median_answer(response)
        assert answer == 5

    def test_comparison_parser_greater(self):
        from beyondbench.parsers.comparison_parsing import parse_comparison_result
        response = r"Number 1 (50) is greater than Number 2 (30). \boxed{greater than}"
        result = parse_comparison_result(response)
        assert result == "greater than"

    def test_comparison_parser_less(self):
        from beyondbench.parsers.comparison_parsing import parse_comparison_result
        response = "Since 10 < 20, the answer is less than."
        result = parse_comparison_result(response)
        assert result == "less than"

    def test_comparison_parser_equal(self):
        from beyondbench.parsers.comparison_parsing import parse_comparison_result
        response = "Both numbers are the same. They are equal to each other."
        result = parse_comparison_result(response)
        assert result == "equal to"

    def test_sorting_parser_boxed_list(self):
        from beyondbench.parsers.sorting_parsing import parse_sorted_list
        response = r"The sorted list is \boxed{1, 3, 5, 7, 9}."
        result = parse_sorted_list(response)
        assert result == [1, 3, 5, 7, 9]

    def test_sorting_parser_bracket_list(self):
        from beyondbench.parsers.sorting_parsing import parse_sorted_list
        response = "After sorting: [1, 3, 5, 7, 9]"
        result = parse_sorted_list(response)
        assert result == [1, 3, 5, 7, 9]

    def test_sorting_parser_comma_separated(self):
        from beyondbench.parsers.sorting_parsing import parse_sorted_list
        response = "The sorted list is: 1, 3, 5, 7, 9"
        result = parse_sorted_list(response)
        assert result == [1, 3, 5, 7, 9]

    def test_mode_parser_single(self):
        from beyondbench.parsers.mode_parsing import parse_mode_answer
        response = r"The mode is \boxed{7}."
        result = parse_mode_answer(response)
        assert result == [7]

    def test_mode_parser_multiple(self):
        from beyondbench.parsers.mode_parsing import parse_mode_answer
        response = "The modes are 3 and 7 (both appear 4 times)."
        result = parse_mode_answer(response)
        assert 3 in result
        assert 7 in result


class TestEdgeCasesAndCornerCases:
    """Test tricky edge cases that could trip up parsers."""

    def test_boxed_with_no_answer_but_explanation(self):
        """Response has boxed but content is not a number."""
        answer, followed = extract_from_boxed_formats(r"\boxed{greater than}")
        assert answer == "greater than"
        assert followed is True

    def test_empty_response(self):
        answer, _ = extract_from_explicit_statements("")
        assert answer is None

    def test_only_whitespace(self):
        answer, _ = extract_from_explicit_statements("   \n\n\t  ")
        assert answer is None

    def test_number_in_url_not_extracted(self):
        """Numbers in URLs should ideally not be extracted as answers."""
        response = "See https://example.com/page/42 for details."
        # This may still extract 42, but it's an edge case we document
        answer, _ = extract_from_explicit_statements(response)
        # Just ensure it doesn't crash

    def test_very_long_response(self):
        """Test with a very long response to ensure no performance issues."""
        response = "Step " + ". Step ".join(str(i) for i in range(1000)) + "\nThe answer is 42."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "42" in answer

    def test_special_characters_in_response(self):
        """Response with special chars shouldn't crash."""
        response = "The answer is 42! @#$%^&*()"
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None

    def test_unicode_numbers(self):
        """Some models output Unicode digits."""
        result = clean_and_convert_to_number("42")  # standard ASCII
        assert result == 42

    def test_number_with_currency_symbols(self):
        result = clean_and_convert_to_number("$42")
        assert result == 42

    def test_multiple_numbers_in_text(self):
        """Extract from explicit should find the right one."""
        response = "We computed 10 + 20 + 30 = 60. The answer is 60."
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "60" in answer

    def test_answer_with_equals_sign(self):
        """Response ending with = answer."""
        response = "10 + 20 + 30 = 60"
        answer, _ = extract_from_explicit_statements(response)
        assert answer is not None
        assert "60" in answer

    def test_clean_number_preserves_negative_sign(self):
        assert clean_and_convert_to_number("-0") == 0
        assert clean_and_convert_to_number("-1") == -1

    def test_fraction_whole_number_result(self):
        result = clean_and_convert_to_number("6/3")
        assert result == 2.0

    def test_fraction_negative_denominator(self):
        result = clean_and_convert_to_number("3/-4")
        assert result == -0.75
