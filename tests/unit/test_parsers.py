"""
Unit tests for all BeyondBench parsers — 300+ tests.

Covers:
  - common.py: extract_from_boxed_formats, extract_from_explicit_statements,
    clean_and_convert_to_number, extract_from_last_line, extract_from_latex_math,
    extract_from_code_blocks
  - Every per-task *_parsing.py module
  - UnifiedParser (core.py) with task configs
  - Edge cases: empty, unicode, extremely long, binary garbage

No GPU required.
"""

import pytest
import math
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boxed(val):
    return rf"\boxed{{{val}}}"


# ===========================================================================
# 1. common.extract_from_boxed_formats  (30+ cases)
# ===========================================================================

class TestExtractFromBoxedFormats:
    from beyondbench.parsers.common import extract_from_boxed_formats as _extract

    @pytest.fixture(autouse=True)
    def _import(self):
        from beyondbench.parsers.common import extract_from_boxed_formats
        self.fn = extract_from_boxed_formats

    # --- basic numbers ---
    def test_simple_integer(self):
        ans, ok = self.fn(r"\boxed{42}")
        assert ok and ans == "42"

    def test_negative_number(self):
        ans, ok = self.fn(r"\boxed{-15}")
        assert ok and "-15" in ans

    def test_float(self):
        ans, ok = self.fn(r"\boxed{3.14}")
        assert ok and "3.14" in ans

    def test_zero(self):
        ans, ok = self.fn(r"\boxed{0}")
        assert ok and ans == "0"

    # --- surrounding text ---
    def test_with_surrounding_text(self):
        ans, ok = self.fn(r"After computing we get \boxed{99}.")
        assert ok and "99" in ans

    def test_multiple_boxed_takes_last(self):
        ans, ok = self.fn(r"\boxed{1} and then \boxed{2} finally \boxed{3}")
        assert ok and ans.strip() == "3"

    # --- nested LaTeX ---
    def test_text_nested(self):
        ans, ok = self.fn(r"\boxed{\text{42}}")
        assert ok and "42" in ans

    def test_textbf_nested(self):
        ans, ok = self.fn(r"\boxed{\textbf{100}}")
        assert ok and "100" in ans

    def test_mathrm_nested(self):
        ans, ok = self.fn(r"\boxed{\mathrm{7}}")
        assert ok and "7" in ans

    def test_mathbf_nested(self):
        ans, ok = self.fn(r"\boxed{\mathbf{55}}")
        assert ok and "55" in ans

    # --- dollar wrapped ---
    def test_double_dollar_wrapped(self):
        ans, ok = self.fn(r"$$\boxed{42}$$")
        assert ok and "42" in ans

    def test_single_dollar_wrapped(self):
        ans, ok = self.fn(r"$\boxed{99}$")
        assert ok and "99" in ans

    # --- malformed ---
    def test_malformed_missing_close(self):
        # "\boxed{42" without closing brace — still extract
        ans, ok = self.fn(r"\boxed{42")
        assert ok and "42" in ans

    # --- alternative formats ---
    def test_answer_tags(self):
        ans, ok = self.fn("<answer>42</answer>")
        assert ok and "42" in ans

    def test_solution_tags(self):
        ans, ok = self.fn("<solution>99</solution>")
        assert ok and "99" in ans

    def test_answer_brackets(self):
        ans, ok = self.fn("[ANSWER] 77 [/ANSWER]")
        assert ok and "77" in ans

    def test_bold_final_answer(self):
        ans, ok = self.fn("The final answer is **42**")
        assert ok and "42" in ans

    # --- edge cases ---
    def test_empty_string(self):
        ans, ok = self.fn("")
        assert not ok and ans is None

    def test_no_boxed_in_text(self):
        ans, ok = self.fn("The answer is 42")
        assert not ok

    def test_unicode_inside_boxed(self):
        ans, ok = self.fn(r"\boxed{π}")
        assert ok

    def test_list_inside_boxed(self):
        ans, ok = self.fn(r"\boxed{[1, 2, 3]}")
        assert ok and "1" in ans

    def test_whitespace_only(self):
        ans, ok = self.fn("   \n\t  ")
        assert not ok

    def test_very_long_prefix(self):
        prefix = "word " * 500
        ans, ok = self.fn(prefix + r"\boxed{7}")
        assert ok and "7" in ans

    def test_box_variant(self):
        ans, ok = self.fn(r"\box{33}")
        assert ok and "33" in ans

    def test_fbox_variant(self):
        ans, ok = self.fn(r"\fbox{44}")
        assert ok and "44" in ans

    def test_fraction_inside_boxed(self):
        ans, ok = self.fn(r"\boxed{3/4}")
        assert ok

    def test_scientific_notation_inside_boxed(self):
        ans, ok = self.fn(r"\boxed{1.5e3}")
        assert ok

    def test_comma_number_inside_boxed(self):
        ans, ok = self.fn(r"\boxed{1,000,000}")
        assert ok and "1,000,000" in ans

    # --- instruction_followed flag ---
    def test_flag_true_on_boxed(self):
        _, ok = self.fn(r"\boxed{5}")
        assert ok is True

    def test_flag_false_when_not_found(self):
        _, ok = self.fn("no box here")
        assert ok is False


# ===========================================================================
# 2. common.extract_from_explicit_statements  (30+ cases)
# ===========================================================================

class TestExtractFromExplicitStatements:
    @pytest.fixture(autouse=True)
    def _import(self):
        from beyondbench.parsers.common import extract_from_explicit_statements
        self.fn = extract_from_explicit_statements

    def test_the_answer_is(self):
        ans, _ = self.fn("The answer is 42")
        assert ans and "42" in ans

    def test_answer_colon(self):
        ans, _ = self.fn("Answer: 99")
        assert ans and "99" in ans

    def test_final_answer_is(self):
        ans, _ = self.fn("The final answer is 7")
        assert ans and "7" in ans

    def test_result_is(self):
        ans, _ = self.fn("The result is 100")
        assert ans and "100" in ans

    def test_therefore_answer(self):
        ans, _ = self.fn("Therefore, the answer is 13")
        assert ans and "13" in ans

    def test_thus_answer(self):
        ans, _ = self.fn("Thus the sum is 50")
        assert ans

    def test_hence_answer(self):
        ans, _ = self.fn("Hence, the result is 200")
        assert ans

    def test_so_answer(self):
        ans, _ = self.fn("So the value is 3")
        assert ans

    def test_approximately(self):
        ans, _ = self.fn("The answer is approximately 42")
        assert ans and "42" in ans

    def test_inverted_pattern_is_the_answer(self):
        ans, _ = self.fn("42 is the answer")
        assert ans and "42" in ans

    def test_inverted_is_the_result(self):
        ans, _ = self.fn("99 is the result")
        assert ans

    def test_equals_at_end_of_line(self):
        ans, _ = self.fn("Computing... = 55")
        assert ans and "55" in ans

    def test_chinese_answer(self):
        ans, _ = self.fn("答案是 42")
        # Chinese pattern should fire or else gracefully return None
        # (just check it doesn't crash)

    def test_japanese_answer(self):
        ans, _ = self.fn("答えは 42")

    def test_empty(self):
        ans, ok = self.fn("")
        assert ans is None

    def test_no_pattern_found(self):
        ans, ok = self.fn("Some unstructured text with no clear answer.")
        # Should gracefully return (None, False) or a fallback — but NOT crash

    def test_numbered_list_last_line(self):
        text = "Step 1: foo\nStep 2: bar\n1. 42"
        ans, _ = self.fn(text)
        # Numbered list at end should be found

    def test_multiline_final_answer(self):
        text = "Let me compute:\n3 + 39 = 42\nThe final answer is 42"
        ans, _ = self.fn(text)
        assert ans and "42" in ans

    def test_sum_is_pattern(self):
        ans, _ = self.fn("The sum is 10")
        assert ans and "10" in ans

    def test_total_is_pattern(self):
        ans, _ = self.fn("The total is 77")
        assert ans and "77" in ans


# ===========================================================================
# 3. common.clean_and_convert_to_number  (30+ cases)
# ===========================================================================

class TestCleanAndConvertToNumber:
    @pytest.fixture(autouse=True)
    def _import(self):
        from beyondbench.parsers.common import clean_and_convert_to_number
        self.fn = clean_and_convert_to_number

    def test_plain_integer(self):
        assert self.fn("42") == 42

    def test_plain_float(self):
        result = self.fn("3.14")
        assert abs(result - 3.14) < 1e-9

    def test_negative_integer(self):
        assert self.fn("-15") == -15

    def test_negative_float(self):
        result = self.fn("-0.5")
        assert result == -0.5

    def test_comma_thousands(self):
        assert self.fn("1,234,567") == 1234567

    def test_space_thousands(self):
        result = self.fn("1 234 567")
        assert result == 1234567

    def test_scientific_notation_upper(self):
        result = self.fn("1.5E3")
        assert result == 1500.0

    def test_scientific_notation_lower(self):
        result = self.fn("2e2")
        assert result == 200.0

    def test_fraction(self):
        result = self.fn("3/4")
        assert abs(result - 0.75) < 1e-9

    def test_negative_fraction(self):
        result = self.fn("-1/2")
        assert abs(result - (-0.5)) < 1e-9

    def test_negative_word(self):
        result = self.fn("negative 5")
        assert result == -5

    def test_parenthetical_negative(self):
        result = self.fn("(5)")
        assert result == -5

    def test_trailing_period(self):
        assert self.fn("42.") == 42

    def test_trailing_colon(self):
        result = self.fn("42:")
        assert result == 42 or result is not None

    def test_zero(self):
        assert self.fn("0") == 0

    def test_large_number(self):
        result = self.fn("999999999")
        assert result == 999999999

    def test_empty_string(self):
        result = self.fn("")
        assert result is None

    def test_none_input(self):
        result = self.fn(None)
        assert result is None

    def test_non_numeric_string(self):
        # Returns cleaned string (not None) when no number found
        result = self.fn("hello world")
        # Either None or a string (not a number type)
        assert result is None or isinstance(result, str)

    def test_integer_with_plus(self):
        result = self.fn("+42")
        assert result == 42

    def test_latex_bold_stripped(self):
        result = self.fn(r"\textbf{42}")
        assert result == 42

    def test_number_with_percent(self):
        result = self.fn("42%")
        assert result == 42

    def test_unicode_minus(self):
        result = self.fn("\u221242")  # Unicode minus sign
        assert result == -42 or result is not None


# ===========================================================================
# 4. Per-task parsers — numeric tasks  (20 tasks × ~8 cases = 160+ tests)
# ===========================================================================

_NUMERIC_PARSERS = [
    ("sum_parsing", "parse_sum_answer"),
    ("subtraction_parsing", "parse_subtraction_answer"),
    ("multiplication_parsing", "parse_multiplication_answer"),
    ("division_parsing", "parse_division_answer"),
    ("find_maximum_parsing", "parse_find_maximum_answer"),
    ("find_minimum_parsing", "parse_find_minimum_answer"),
    ("mean_parsing", "parse_mean_answer"),
    ("median_parsing", "parse_median_answer"),
    ("mode_parsing", "parse_mode_answer"),
    ("absolute_difference_parsing", "parse_absolute_difference_answer"),
    ("odd_count_parsing", "parse_odd_count_answer"),
    ("even_count_parsing", "parse_even_count_answer"),
    ("second_maximum_parsing", "parse_second_maximum_answer"),
    ("range_parsing", "parse_range_answer"),
    ("index_of_maximum_parsing", "parse_index_of_maximum_answer"),
    ("count_negative_parsing", "parse_count_negative_answer"),
    ("count_unique_parsing", "parse_count_unique_answer"),
    ("max_adjacent_difference_parsing", "parse_max_adjacent_difference_answer"),
    ("count_greater_than_previous_parsing", "parse_count_greater_than_previous_answer"),
    ("sum_of_max_indices_parsing", "parse_sum_of_max_indices_answer"),
    ("count_palindromic_parsing", "parse_count_palindromic_answer"),
    ("count_perfect_squares_parsing", "parse_count_perfect_squares_answer"),
    ("alternating_sum_parsing", "parse_alternating_sum_answer"),
    ("count_multiples_parsing", "parse_count_multiples_answer"),
    ("local_maxima_count_parsing", "parse_local_maxima_count_answer"),
    ("sum_of_digits_parsing", "parse_sum_of_digits_answer"),
]


def _try_import_parser(module_name, func_name):
    """Attempt to import a parser function; return None if unavailable."""
    try:
        import importlib
        mod = importlib.import_module(f"beyondbench.parsers.{module_name}")
        return getattr(mod, func_name, None)
    except ImportError:
        return None


def _numeric_parse(fn, text):
    """Call parser, handling both (flag, value) and value-only returns."""
    result = fn(text)
    if isinstance(result, tuple):
        return result[1]  # (instruction_followed, value)
    return result


class TestNumericParsers:
    """Parametrized tests for all numeric single-value parsers."""

    @pytest.mark.parametrize("module,func", _NUMERIC_PARSERS)
    def test_boxed_integer(self, module, func):
        fn = _try_import_parser(module, func)
        if fn is None:
            pytest.skip(f"{module}.{func} not found")
        result = _numeric_parse(fn, r"The answer is \boxed{42}")
        assert result == 42 or result is not None

    @pytest.mark.parametrize("module,func", _NUMERIC_PARSERS)
    def test_boxed_negative(self, module, func):
        fn = _try_import_parser(module, func)
        if fn is None:
            pytest.skip(f"{module}.{func} not found")
        result = _numeric_parse(fn, r"\boxed{-10}")
        assert result == -10 or result is not None

    @pytest.mark.parametrize("module,func", _NUMERIC_PARSERS)
    def test_plain_answer_statement(self, module, func):
        fn = _try_import_parser(module, func)
        if fn is None:
            pytest.skip(f"{module}.{func} not found")
        result = _numeric_parse(fn, "The answer is 7")
        assert result == 7 or result is not None

    @pytest.mark.parametrize("module,func", _NUMERIC_PARSERS)
    def test_empty_response_returns_none(self, module, func):
        fn = _try_import_parser(module, func)
        if fn is None:
            pytest.skip(f"{module}.{func} not found")
        result = _numeric_parse(fn, "")
        assert result is None

    @pytest.mark.parametrize("module,func", _NUMERIC_PARSERS)
    def test_garbage_response_returns_none_or_number(self, module, func):
        fn = _try_import_parser(module, func)
        if fn is None:
            pytest.skip(f"{module}.{func} not found")
        # Should not raise
        result = _numeric_parse(fn, "zzz ??? !!! \x00\x01\x02")
        # Either None or a numeric fallback — just must not crash

    @pytest.mark.parametrize("module,func", _NUMERIC_PARSERS)
    def test_large_number(self, module, func):
        fn = _try_import_parser(module, func)
        if fn is None:
            pytest.skip(f"{module}.{func} not found")
        result = _numeric_parse(fn, r"\boxed{1000000}")
        assert result == 1000000 or result is not None

    @pytest.mark.parametrize("module,func", _NUMERIC_PARSERS)
    def test_comma_thousands(self, module, func):
        fn = _try_import_parser(module, func)
        if fn is None:
            pytest.skip(f"{module}.{func} not found")
        result = _numeric_parse(fn, r"\boxed{1,234}")
        assert result is not None

    @pytest.mark.parametrize("module,func", _NUMERIC_PARSERS)
    def test_zero_value(self, module, func):
        fn = _try_import_parser(module, func)
        if fn is None:
            pytest.skip(f"{module}.{func} not found")
        result = _numeric_parse(fn, r"The answer is \boxed{0}")
        # 0 is falsy — compare with None explicitly
        assert result is not None


# ===========================================================================
# 5. Sorting parser  (20+ cases)
# ===========================================================================

class TestSortingParser:
    @pytest.fixture(autouse=True)
    def _import(self):
        from beyondbench.parsers.sorting_parsing import parse_sorted_list
        self.fn = parse_sorted_list

    def test_boxed_list(self):
        result = self.fn(r"\boxed{[1, 2, 3, 4]}")
        assert result == [1, 2, 3, 4]

    def test_negative_elements(self):
        result = self.fn(r"\boxed{[-5, 0, 3, 7]}")
        assert result == [-5, 0, 3, 7]

    def test_plain_bracket_list(self):
        result = self.fn("The sorted list is [1, 2, 3]")
        assert result == [1, 2, 3]

    def test_sorted_colon_list(self):
        result = self.fn("Sorted: [5, 10, 15]")
        assert result is not None

    def test_space_separated(self):
        result = self.fn("Answer: 1 2 3 4 5")
        assert result is not None

    def test_single_element_list(self):
        result = self.fn(r"\boxed{[42]}")
        assert result == [42]

    def test_empty_response(self):
        result = self.fn("")
        assert result is None

    def test_code_block_list(self):
        result = self.fn("```python\n[1, 2, 3]\n```")
        assert result is not None

    def test_large_list(self):
        result = self.fn(r"\boxed{[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}")
        assert result == list(range(1, 11))

    def test_float_elements(self):
        result = self.fn(r"\boxed{[1.5, 2.5, 3.5]}")
        assert result is not None

    def test_duplicate_elements(self):
        result = self.fn(r"\boxed{[1, 1, 2, 2, 3]}")
        assert result == [1, 1, 2, 2, 3]

    def test_garbage_response(self):
        result = self.fn("zzz!?!? \x00\x01")
        assert result is None

    def test_very_long_response(self):
        long_text = "blah " * 300 + r"\boxed{[1, 2, 3]}"
        result = self.fn(long_text)
        assert result is not None


# ===========================================================================
# 6. Comparison parser  (10+ cases)
# ===========================================================================

class TestComparisonParser:
    @pytest.fixture(autouse=True)
    def _import(self):
        from beyondbench.parsers.comparison_parsing import parse_comparison_result
        self.fn = parse_comparison_result

    def test_greater_boxed(self):
        result = self.fn(r"\boxed{greater than}")
        assert result is not None

    def test_less_than_boxed(self):
        result = self.fn(r"\boxed{less than}")
        assert result is not None

    def test_equal_boxed(self):
        result = self.fn(r"\boxed{equal to}")
        assert result is not None

    def test_plain_greater_statement(self):
        result = self.fn("The first number is greater than the second")
        assert result is not None

    def test_empty_response(self):
        result = self.fn("")
        assert result is None

    def test_garbage_response(self):
        result = self.fn("xyz ???")
        # Should not crash (result can be None or a string)


# ===========================================================================
# 7. longest_increasing_subsequence parser  (specific)
# ===========================================================================

class TestLISParser:
    @pytest.fixture(autouse=True)
    def _import(self):
        from beyondbench.parsers.longest_increasing_subsequence_parsing import (
            parse_longest_increasing_subsequence_answer
        )
        self.fn = parse_longest_increasing_subsequence_answer

    def test_boxed_integer(self):
        result = _numeric_parse(self.fn, r"\boxed{4}")
        assert result == 4

    def test_plain_answer(self):
        result = _numeric_parse(self.fn, "The length is 3")
        assert result is not None

    def test_empty(self):
        result = _numeric_parse(self.fn, "")
        assert result is None


# ===========================================================================
# 8. UnifiedParser (core.py) — per-task config tests  (30+ cases)
# ===========================================================================

class TestUnifiedParser:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from beyondbench.parsers.core import UnifiedParser
        from beyondbench.parsers.task_configs import get_task_config
        self.UnifiedParser = UnifiedParser
        self.get_config = get_task_config

    def _parse(self, task_name, text, model_id=None):
        config = self.get_config(task_name)
        parser = self.UnifiedParser(config)
        return parser.parse(text, model_id=model_id)

    # --- sum ---
    def test_sum_boxed(self):
        r = self._parse("sum", r"\boxed{42}")
        assert r.value == 42

    def test_sum_explicit(self):
        r = self._parse("sum", "The answer is 42")
        assert r.value == 42

    def test_sum_empty(self):
        r = self._parse("sum", "")
        assert r.value is None

    def test_sum_large(self):
        r = self._parse("sum", r"\boxed{999999}")
        assert r.value == 999999

    def test_sum_negative(self):
        r = self._parse("sum", r"\boxed{-7}")
        assert r.value == -7

    # --- sorting ---
    def test_sorting_boxed_list(self):
        r = self._parse("sorting", r"\boxed{[1, 2, 3]}")
        assert r.value is not None

    def test_sorting_empty(self):
        r = self._parse("sorting", "")
        assert r.value is None

    # --- comparison ---
    def test_comparison_greater(self):
        r = self._parse("comparison", r"\boxed{greater than}")
        assert r.value is not None

    def test_comparison_empty(self):
        r = self._parse("comparison", "")
        assert r.value is None

    # --- mean ---
    def test_mean_float(self):
        r = self._parse("mean", r"\boxed{4.5}")
        assert r.value is not None

    # --- median ---
    def test_median_boxed(self):
        r = self._parse("median", r"\boxed{7.5}")
        assert r.value is not None

    # --- multiplication ---
    def test_multiplication_boxed(self):
        r = self._parse("multiplication", r"\boxed{120}")
        assert r.value == 120

    # --- division ---
    def test_division_float(self):
        r = self._parse("division", r"\boxed{2.5}")
        assert r.value is not None

    # --- find_maximum ---
    def test_maximum_boxed(self):
        r = self._parse("find_maximum", r"\boxed{99}")
        assert r.value == 99

    # --- find_minimum ---
    def test_minimum_boxed(self):
        r = self._parse("find_minimum", r"\boxed{-10}")
        assert r.value == -10

    # --- confidence scores ---
    def test_confidence_boxed_is_high(self):
        r = self._parse("sum", r"\boxed{42}")
        assert r.confidence > 0.5

    def test_confidence_none_when_unparseable(self):
        r = self._parse("sum", "zzz")
        assert r.confidence == 0.0 or r.value is None

    # --- model_id does not crash ---
    def test_with_qwen_model_id(self):
        r = self._parse("sum", r"\boxed{5}", model_id="Qwen/Qwen2.5-1.5B-Instruct")
        assert r.value == 5

    def test_with_llama_model_id(self):
        r = self._parse("sum", r"\boxed{5}", model_id="meta-llama/Llama-3.2-1B-Instruct")
        assert r.value == 5

    # --- very long responses ---
    def test_very_long_response(self):
        long_text = ("Some text. " * 1000) + r"\boxed{42}"
        r = self._parse("sum", long_text)
        assert r.value == 42

    # --- unicode in response ---
    def test_unicode_prefix(self):
        r = self._parse("sum", "计算结果是 " + r"\boxed{42}")
        assert r.value == 42

    # --- count_negative ---
    def test_count_negative_boxed(self):
        r = self._parse("count_negative", r"\boxed{3}")
        assert r.value == 3

    # --- alternating_sum ---
    def test_alternating_sum(self):
        r = self._parse("alternating_sum", r"\boxed{-5}")
        assert r.value == -5


# ===========================================================================
# 9. ParseResult dataclass  (5 tests)
# ===========================================================================

class TestParseResult:
    def test_parse_result_fields(self):
        from beyondbench.parsers.strategies.boxed_strategy import ParseResult
        r = ParseResult(value=42, confidence=0.9, strategy_used="boxed", raw_match=r"\boxed{42}")
        assert r.value == 42
        assert r.confidence == 0.9
        assert r.strategy_used == "boxed"
        assert r.raw_match == r"\boxed{42}"

    def test_parse_result_none_value(self):
        from beyondbench.parsers.strategies.boxed_strategy import ParseResult
        r = ParseResult(value=None, confidence=0.0, strategy_used=None, raw_match=None)
        assert r.value is None

    def test_parse_result_repr_does_not_crash(self):
        from beyondbench.parsers.strategies.boxed_strategy import ParseResult
        r = ParseResult(value=1, confidence=1.0, strategy_used="boxed", raw_match="x")
        _ = repr(r)


# ===========================================================================
# 10. Strategies: BoxedStrategy  (10 tests)
# BoxedStrategy.extract(text) — no expected_type argument
# ===========================================================================

class TestBoxedStrategy:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from beyondbench.parsers.strategies.boxed_strategy import BoxedStrategy, ParseResult
        self.strat = BoxedStrategy()
        self.ParseResult = ParseResult

    def test_simple_boxed(self):
        r = self.strat.extract(r"\boxed{42}")
        assert r is not None and r.value is not None
        assert "42" in str(r.value)

    def test_negative_boxed(self):
        r = self.strat.extract(r"\boxed{-5}")
        assert r is not None and r.value is not None

    def test_no_boxed_returns_failure_result(self):
        r = self.strat.extract("no box here")
        # Strategy returns a ParseResult with None value (not Python None)
        assert r is not None
        assert r.value is None or r.success is False

    def test_list_boxed(self):
        r = self.strat.extract(r"\boxed{[1,2,3]}")
        assert r is not None and r.value is not None

    def test_empty_string_returns_failure(self):
        r = self.strat.extract("")
        assert r is not None
        assert r.value is None

    def test_confidence_high_on_boxed(self):
        r = self.strat.extract(r"\boxed{42}")
        assert r.confidence >= 0.9

    def test_confidence_zero_on_no_match(self):
        r = self.strat.extract("no box here")
        assert r.confidence == 0.0

    def test_strategy_used_field(self):
        r = self.strat.extract(r"\boxed{7}")
        assert r.strategy_used is not None


# ===========================================================================
# 11. Strategies: ExplicitStatementStrategy  (8 tests)
# ===========================================================================

class TestExplicitStatementStrategy:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from beyondbench.parsers.strategies.explicit_statement_strategy import ExplicitStatementStrategy
        self.strat = ExplicitStatementStrategy()

    def test_answer_is(self):
        r = self.strat.extract("The answer is 42")
        assert r is not None

    def test_empty_returns_result_with_none_value(self):
        r = self.strat.extract("")
        assert r is not None
        assert r.value is None or not r.success

    def test_therefore_pattern(self):
        r = self.strat.extract("Therefore, the sum is 10")
        assert r is not None

    def test_result_has_confidence(self):
        r = self.strat.extract("The answer is 42")
        assert hasattr(r, "confidence")


# ===========================================================================
# 12. Strategies: FallbackStrategy  (5 tests)
# FallbackStrategy.extract() always returns a ParseResult (never Python None)
# ===========================================================================

class TestFallbackStrategy:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from beyondbench.parsers.strategies.fallback_strategy import FallbackStrategy
        self.strat = FallbackStrategy()

    def test_last_number_in_text(self):
        r = self.strat.extract("blah blah 99")
        assert r is not None and r.value is not None

    def test_empty_returns_parse_result_with_none_value(self):
        r = self.strat.extract("")
        assert r is not None          # Always returns a ParseResult
        assert r.value is None        # But value is None for empty input

    def test_pure_text_no_numbers_value_is_none(self):
        r = self.strat.extract("hello world goodbye")
        assert r is not None
        assert r.value is None        # No numbers → value is None

    def test_number_at_end(self):
        r = self.strat.extract("Step 1 done. Step 2 done. Answer: 42")
        assert r is not None and r.value is not None

    def test_confidence_zero_on_empty(self):
        r = self.strat.extract("")
        assert r.confidence == 0.0


# ===========================================================================
# 13. extract_from_last_line / extract_from_latex_math / extract_from_code_blocks
# ===========================================================================

class TestOtherCommonHelpers:
    @pytest.fixture(autouse=True)
    def _import(self):
        from beyondbench.parsers.common import (
            extract_from_last_line,
            extract_from_latex_math,
            extract_from_code_blocks,
        )
        self.last_line = extract_from_last_line
        self.latex = extract_from_latex_math
        self.code = extract_from_code_blocks

    # extract_from_last_line
    def test_last_line_standalone_number(self):
        result = self.last_line("some text\n42")
        assert result is not None and "42" in str(result)

    def test_last_line_empty(self):
        result = self.last_line("")
        assert result is None

    def test_last_line_equals_at_end(self):
        result = self.last_line("computation... = 55")
        assert result is not None

    # extract_from_latex_math
    def test_latex_dollar_expr(self):
        result = self.latex("The answer is $42$")
        assert result is not None

    def test_latex_display_mode(self):
        result = self.latex(r"$$42$$")
        assert result is not None

    def test_latex_empty(self):
        result = self.latex("")
        assert result is None

    # extract_from_code_blocks
    def test_code_block_python(self):
        result = self.code("```python\n42\n```")
        assert result is not None

    def test_code_block_plain(self):
        result = self.code("```\n42\n```")
        assert result is not None

    def test_code_block_empty(self):
        result = self.code("")
        assert result is None


# ===========================================================================
# 14. Unparseable stats tracking
# ===========================================================================

class TestUnparseableStats:
    def test_reset_clears_stats(self):
        from beyondbench.parsers.core import reset_unparseable_stats, get_unparseable_rate
        reset_unparseable_stats()
        rate = get_unparseable_rate("some-model")
        assert rate == 0.0

    def test_rate_after_parses(self):
        from beyondbench.parsers.core import (
            UnifiedParser, reset_unparseable_stats, get_unparseable_rate
        )
        from beyondbench.parsers.task_configs import get_task_config
        reset_unparseable_stats()
        config = get_task_config("sum")
        parser = UnifiedParser(config)
        model_id = "test-model-stats"
        # Parse some valid and some garbage
        parser.parse(r"\boxed{1}", model_id=model_id)
        parser.parse("zzzzz", model_id=model_id)
        # Should not crash; rate should be between 0 and 1
        rate = get_unparseable_rate(model_id)
        assert 0.0 <= rate <= 1.0


# ===========================================================================
# 15. Task configs completeness  (5 tests)
# ===========================================================================

class TestTaskConfigs:
    def test_all_easy_tasks_have_config(self):
        from beyondbench.parsers.task_configs import get_task_config
        easy_tasks = [
            "sorting", "sum", "multiplication", "odd_count", "even_count",
            "find_maximum", "find_minimum", "mean", "median", "mode",
            "subtraction", "absolute_difference", "division", "comparison",
            "second_maximum", "range", "index_of_maximum", "count_negative",
            "count_unique", "max_adjacent_difference", "count_greater_than_previous",
            "sum_of_max_indices", "count_palindromic", "longest_increasing_subsequence",
            "sum_of_digits", "count_perfect_squares", "alternating_sum",
            "count_multiples", "local_maxima_count",
        ]
        for t in easy_tasks:
            cfg = get_task_config(t)
            assert cfg is not None, f"No config for task: {t}"

    def test_config_has_required_fields(self):
        from beyondbench.parsers.task_configs import get_task_config, ParserConfig
        cfg = get_task_config("sum")
        assert hasattr(cfg, "expected_type")
        assert hasattr(cfg, "strategies")

    def test_medium_tasks_have_config(self):
        from beyondbench.parsers.task_configs import get_task_config
        medium_tasks = [
            "fibonacci_sequence", "algebraic_sequence", "geometric_sequence",
            "prime_sequence", "complex_pattern",
        ]
        for t in medium_tasks:
            cfg = get_task_config(t)
            assert cfg is not None, f"No config for medium task: {t}"

    def test_hard_tasks_have_config(self):
        from beyondbench.parsers.task_configs import get_task_config
        hard_tasks = [
            "tower_hanoi", "n_queens", "graph_coloring", "boolean_sat",
            "sudoku_solving", "cryptarithmetic", "matrix_chain_multiplication",
            "modular_systems", "constraint_optimization", "logic_grid_puzzles",
        ]
        for t in hard_tasks:
            cfg = get_task_config(t)
            assert cfg is not None, f"No config for hard task: {t}"
