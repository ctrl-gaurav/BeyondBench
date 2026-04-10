"""
Comprehensive tests for the Phase 1 UnifiedParser.

Tests:
- Each strategy independently
- Model adapters
- Per-task config lookup
- End-to-end parsing for all task types
- Regression: parse success rate must not decrease

Run:
    pytest tests/test_unified_parser.py -v --timeout=60
"""

import pytest
from beyondbench.parsers.core import UnifiedParser, parse_response
from beyondbench.parsers.strategies.boxed_strategy import BoxedStrategy, ParseResult
from beyondbench.parsers.strategies.explicit_statement_strategy import ExplicitStatementStrategy
from beyondbench.parsers.strategies.code_block_strategy import CodeBlockStrategy
from beyondbench.parsers.strategies.latex_math_strategy import LatexMathStrategy
from beyondbench.parsers.strategies.json_strategy import JsonStrategy
from beyondbench.parsers.strategies.list_strategy import ListStrategy
from beyondbench.parsers.strategies.grid_strategy import GridStrategy
from beyondbench.parsers.strategies.comparison_strategy import ComparisonStrategy
from beyondbench.parsers.strategies.sequence_strategy import SequenceStrategy
from beyondbench.parsers.strategies.fallback_strategy import FallbackStrategy
from beyondbench.parsers.model_adapters import (
    QwenAdapter, LlamaAdapter, PhiAdapter, MistralAdapter, GemmaAdapter, APIAdapter,
    get_adapter,
)
from beyondbench.parsers.task_configs import get_task_config, TASK_PARSER_CONFIGS


# ============================================================================
# ParseResult basic tests
# ============================================================================

def test_parse_result_success():
    r = ParseResult(value="42", confidence=0.9, strategy_used="boxed")
    assert r.success is True


def test_parse_result_failure():
    r = ParseResult(value=None, confidence=0.0, strategy_used="boxed")
    assert r.success is False


# ============================================================================
# BoxedStrategy tests
# ============================================================================

class TestBoxedStrategy:
    strategy = BoxedStrategy()

    @pytest.mark.parametrize("text,expected", [
        (r"\boxed{42}", "42"),
        (r"\boxed{-17}", "-17"),
        (r"\boxed{3.14}", "3.14"),
        (r"The answer is \boxed{100}.", "100"),
        (r"$$\boxed{42}$$", "42"),
        (r"$\boxed{42}$", "42"),
        (r"\boxed{\text{42}}", "42"),
        (r"\boxed{\textbf{100}}", "100"),
        (r"\boxed{\mathrm{99}}", "99"),
        (r"\boxed{\mathbf{77}}", "77"),
        (r"First \boxed{10}, then \boxed{42}", "42"),   # Takes LAST
        (r"\box{15}", "15"),
        (r"\fbox{33}", "33"),
        (r"<answer>55</answer>", "55"),
        (r"<solution>  77  </solution>", "77"),
        ("[ANSWER] 88 [/ANSWER]", "88"),
        (r"The final answer is **42**", "42"),
        ("[boxed{42}]", "42"),
    ])
    def test_extracts_boxed(self, text, expected):
        result = self.strategy.extract(text)
        assert result.success, f"Failed to parse: {text!r}"
        assert result.value == expected, f"Expected {expected!r}, got {result.value!r} from {text!r}"

    def test_malformed_boxed(self):
        result = self.strategy.extract(r"\boxed{42")
        assert result.success
        assert "42" in result.value

    def test_no_match(self):
        result = self.strategy.extract("The answer is forty-two")
        assert not result.success

    def test_confidence_high(self):
        result = self.strategy.extract(r"\boxed{42}")
        assert result.confidence >= 0.90

    def test_takes_last_match(self):
        result = self.strategy.extract(r"\boxed{10} and later \boxed{42}")
        assert result.value == "42"


# ============================================================================
# ExplicitStatementStrategy tests
# ============================================================================

class TestExplicitStatementStrategy:
    strategy = ExplicitStatementStrategy()

    @pytest.mark.parametrize("text,expected_substr", [
        ("The answer is 42", "42"),
        ("Answer: 100", "100"),
        ("The final answer is -17", "-17"),
        ("Therefore, the answer is 99", "99"),
        ("Thus, the result is 55", "55"),
        ("Hence, the value is 7", "7"),
        ("The total is 1000", "1000"),
        ("The sum is 42", "42"),
        ("The result is 3.14", "3.14"),
        ("The answer is approximately 42", "42"),
        ("42 is the answer", "42"),
        ("The output is 77", "77"),
        ("= 99", "99"),
    ])
    def test_extracts_explicit(self, text, expected_substr):
        result = self.strategy.extract(text)
        assert result.success, f"Failed to parse: {text!r}"
        assert expected_substr in str(result.value)

    def test_no_match(self):
        result = self.strategy.extract("The weather is nice today")
        # May or may not find something — just verify it doesn't crash
        assert isinstance(result, ParseResult)

    def test_chinese_pattern(self):
        result = self.strategy.extract("答案是42")
        # May not match all Chinese patterns but should not crash
        assert isinstance(result, ParseResult)

    def test_multilingual_graceful(self):
        result = self.strategy.extract("答案：42")
        assert isinstance(result, ParseResult)


# ============================================================================
# CodeBlockStrategy tests
# ============================================================================

class TestCodeBlockStrategy:
    strategy = CodeBlockStrategy()

    def test_python_print(self):
        text = "```python\nprint(42)\n```"
        result = self.strategy.extract(text)
        assert result.success
        assert "42" in str(result.value)

    def test_return_statement(self):
        text = "```python\nreturn 100\n```"
        result = self.strategy.extract(text)
        assert result.success
        assert "100" in str(result.value)

    def test_assignment(self):
        text = "```python\nanswer = 99\n```"
        result = self.strategy.extract(text)
        assert result.success
        assert "99" in str(result.value)

    def test_standalone_number_block(self):
        text = "```\n42\n```"
        result = self.strategy.extract(text)
        assert result.success
        assert "42" in str(result.value)

    def test_no_code_block(self):
        result = self.strategy.extract("The answer is 42")
        # Should not find anything from a code block
        assert isinstance(result, ParseResult)

    def test_print_string(self):
        text = '```python\nprint("42")\n```'
        result = self.strategy.extract(text)
        assert result.success


# ============================================================================
# LatexMathStrategy tests
# ============================================================================

class TestLatexMathStrategy:
    strategy = LatexMathStrategy()

    @pytest.mark.parametrize("text", [
        "$42$",
        "$$42$$",
        r"\(42\)",
        r"\[42\]",
    ])
    def test_extracts_latex(self, text):
        result = self.strategy.extract(text)
        assert result.success
        assert "42" in str(result.value)

    def test_no_match(self):
        result = self.strategy.extract("The answer is 42")
        assert isinstance(result, ParseResult)


# ============================================================================
# JsonStrategy tests
# ============================================================================

class TestJsonStrategy:
    strategy = JsonStrategy()

    def test_json_answer_key(self):
        result = self.strategy.extract('{"answer": 42}')
        assert result.success
        assert "42" in str(result.value)

    def test_json_result_key(self):
        result = self.strategy.extract('{"result": 100}')
        assert result.success
        assert "100" in str(result.value)

    def test_json_value_key(self):
        result = self.strategy.extract('{"value": -17}')
        assert result.success
        assert "-17" in str(result.value)

    def test_json_array(self):
        result = self.strategy.extract("[1, 2, 3, 4]")
        assert isinstance(result, ParseResult)

    def test_no_json(self):
        result = self.strategy.extract("The answer is 42")
        assert isinstance(result, ParseResult)


# ============================================================================
# ListStrategy tests
# ============================================================================

class TestListStrategy:
    strategy = ListStrategy()

    @pytest.mark.parametrize("text,min_length", [
        ("[1, 2, 3, 4, 5]", 5),
        ("(1, 2, 3)", 3),
        ("-5, 3, 7, 12", 4),
        ("[10, 20, 30, 40, 50, 60]", 6),
    ])
    def test_extracts_list(self, text, min_length):
        result = self.strategy.extract(text, expected_type="list")
        assert result.success, f"Failed to parse list from: {text!r}"
        import ast
        parsed = ast.literal_eval(result.value)
        assert isinstance(parsed, list)
        assert len(parsed) >= min_length

    def test_sorted_list_explicit(self):
        result = self.strategy.extract(
            "The sorted list is: [1, 2, 3, 4, 5]",
            expected_type="list"
        )
        assert result.success

    def test_no_list(self):
        result = self.strategy.extract("The answer is 42", expected_type="list")
        assert isinstance(result, ParseResult)

    def test_wrong_type_skips(self):
        result = self.strategy.extract("[1, 2, 3]", expected_type="int")
        assert not result.success


# ============================================================================
# GridStrategy tests
# ============================================================================

class TestGridStrategy:
    strategy = GridStrategy()

    def test_python_list_of_lists(self):
        text = "[[1,2,3],[4,5,6],[7,8,9]]"
        result = self.strategy.extract(text, expected_type="grid")
        assert result.success

    def test_multiline_grid(self):
        text = "1 2 3\n4 5 6\n7 8 9"
        result = self.strategy.extract(text, expected_type="grid")
        assert result.success

    def test_markdown_table(self):
        text = "| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n| 7 | 8 | 9 |"
        result = self.strategy.extract(text, expected_type="grid")
        assert result.success

    def test_wrong_type_skips(self):
        result = self.strategy.extract("[[1,2],[3,4]]", expected_type="int")
        assert not result.success


# ============================================================================
# ComparisonStrategy tests
# ============================================================================

class TestComparisonStrategy:
    strategy = ComparisonStrategy()

    @pytest.mark.parametrize("text,expected", [
        ("The answer is greater than", "greater than"),
        ("Number 1 is greater than Number 2", "greater than"),
        ("The result is less than", "less than"),
        ("Number 1 is less than Number 2", "less than"),
        ("They are equal to each other", "equal to"),
        ("42 > 37", "greater than"),
        ("5 < 10", "less than"),
        ("Number 1 is larger", "greater than"),
        ("Number 1 is smaller", "less than"),
        ("The first is bigger", "greater than"),
        ("Number 1 is more", "greater than"),
        ("Number 1 is fewer", "less than"),
    ])
    def test_extracts_comparison(self, text, expected):
        result = self.strategy.extract(text, expected_type="comparison")
        assert result.success, f"Failed to parse: {text!r}"
        assert result.value == expected, f"Expected {expected!r}, got {result.value!r}"

    def test_wrong_type_skips(self):
        result = self.strategy.extract("greater than", expected_type="int")
        assert not result.success


# ============================================================================
# SequenceStrategy tests
# ============================================================================

class TestSequenceStrategy:
    strategy = SequenceStrategy()

    @pytest.mark.parametrize("text,expected_substr", [
        ("The next term is 144", "144"),
        ("The next number is 21", "21"),
        ("The following term is 89", "89"),
        ("Therefore, the next term is 377", "377"),
        ("Continuing the pattern, 55", "55"),
    ])
    def test_extracts_sequence(self, text, expected_substr):
        result = self.strategy.extract(text)
        assert result.success, f"Failed to parse: {text!r}"
        assert expected_substr in str(result.value)

    def test_no_match(self):
        result = self.strategy.extract("The weather is nice")
        assert isinstance(result, ParseResult)


# ============================================================================
# FallbackStrategy tests
# ============================================================================

class TestFallbackStrategy:
    strategy = FallbackStrategy()

    def test_last_line_number(self):
        result = self.strategy.extract("Some text\n42")
        assert result.success
        assert "42" in str(result.value)

    def test_last_number(self):
        result = self.strategy.extract("Numbers: 1, 2, 3, 42")
        assert result.success
        assert "42" in str(result.value)

    def test_empty_text(self):
        result = self.strategy.extract("")
        assert not result.success

    def test_low_confidence(self):
        result = self.strategy.extract("Some text\n42")
        assert result.confidence < 0.90


# ============================================================================
# Model Adapters tests
# ============================================================================

class TestModelAdapters:

    def test_qwen_strips_tokens(self):
        adapter = QwenAdapter()
        text = "The answer is 42<|im_end|>"
        cleaned = adapter.normalize(text)
        assert "<|im_end|>" not in cleaned
        assert "42" in cleaned

    def test_qwen_strips_start_tokens(self):
        adapter = QwenAdapter()
        text = "<|im_start|>assistant\nThe answer is 42<|im_end|>"
        cleaned = adapter.normalize(text)
        assert "<|im_start|>" not in cleaned
        assert "42" in cleaned

    def test_llama_strips_inst(self):
        adapter = LlamaAdapter()
        text = "[INST] What is 2+2? [/INST] The answer is 4"
        cleaned = adapter.normalize(text)
        assert "[INST]" not in cleaned
        assert "4" in cleaned

    def test_phi_strips_end_token(self):
        adapter = PhiAdapter()
        text = "42<|end|>"
        cleaned = adapter.normalize(text)
        assert "<|end|>" not in cleaned
        assert "42" in cleaned

    def test_mistral_strips_inst(self):
        adapter = MistralAdapter()
        text = "[INST] question [/INST] The answer is 42"
        cleaned = adapter.normalize(text)
        assert "[INST]" not in cleaned
        assert "42" in cleaned

    def test_gemma_strips_turn(self):
        adapter = GemmaAdapter()
        text = "The answer is 42<end_of_turn>"
        cleaned = adapter.normalize(text)
        assert "<end_of_turn>" not in cleaned
        assert "42" in cleaned

    def test_api_strips_role(self):
        adapter = APIAdapter()
        text = "assistant: The answer is 42"
        cleaned = adapter.normalize(text)
        assert cleaned.startswith("The")

    def test_base_passthrough(self):
        from beyondbench.parsers.model_adapters import BaseAdapter
        adapter = BaseAdapter()
        assert adapter.normalize("hello") == "hello"

    def test_empty_string(self):
        adapter = QwenAdapter()
        assert adapter.normalize("") == ""
        assert adapter.normalize(None) == ""


class TestGetAdapter:
    @pytest.mark.parametrize("model_id,expected_cls", [
        ("Qwen/Qwen2.5-1.5B-Instruct", QwenAdapter),
        ("qwen2.5-7b", QwenAdapter),
        ("meta-llama/Llama-3.2-3B-Instruct", LlamaAdapter),
        ("llama-2-7b", LlamaAdapter),
        ("microsoft/Phi-3.5-mini-instruct", PhiAdapter),
        ("phi-2", PhiAdapter),
        ("mistralai/Mistral-7B-Instruct-v0.3", MistralAdapter),
        ("google/gemma-2-9b", GemmaAdapter),
        ("gpt-4o", APIAdapter),
        ("claude-3-opus", APIAdapter),
    ])
    def test_auto_detect(self, model_id, expected_cls):
        adapter = get_adapter(model_id)
        assert isinstance(adapter, expected_cls), \
            f"Expected {expected_cls.__name__} for {model_id!r}, got {type(adapter).__name__}"

    def test_unknown_model_returns_base(self):
        from beyondbench.parsers.model_adapters import BaseAdapter
        adapter = get_adapter("some-unknown-model-xyz")
        assert isinstance(adapter, BaseAdapter)

    def test_none_returns_base(self):
        from beyondbench.parsers.model_adapters import BaseAdapter
        adapter = get_adapter(None)
        assert isinstance(adapter, BaseAdapter)


# ============================================================================
# Task Config tests
# ============================================================================

class TestTaskConfigs:

    @pytest.mark.parametrize("task_name,expected_type", [
        ("sum", "int"),
        ("sorting", "list"),
        ("comparison", "comparison"),
        ("division", "float"),
        ("fibonacci_sequence", "int"),
        ("sudoku_solving", "grid"),
    ])
    def test_known_task_configs(self, task_name, expected_type):
        config = get_task_config(task_name)
        assert config.expected_type == expected_type

    def test_all_tasks_have_strategies(self):
        for task_name, config in TASK_PARSER_CONFIGS.items():
            assert len(config.strategies) >= 2, \
                f"Task {task_name!r} has fewer than 2 strategies"

    def test_unknown_task_returns_fallback(self):
        config = get_task_config("nonexistent_task_xyz")
        assert config is not None
        assert len(config.strategies) > 0

    def test_all_strategies_known(self):
        known = {"boxed", "explicit_statement", "code_block", "latex_math",
                 "json", "list", "grid", "comparison", "sequence", "fallback"}
        for task_name, config in TASK_PARSER_CONFIGS.items():
            for s in config.strategies:
                assert s in known, f"Task {task_name!r} uses unknown strategy {s!r}"


# ============================================================================
# UnifiedParser end-to-end tests
# ============================================================================

class TestUnifiedParser:

    # --- Numeric tasks ---

    @pytest.mark.parametrize("response,expected", [
        (r"The answer is \boxed{42}", 42),
        (r"$$\boxed{100}$$", 100),
        ("The sum is 99", 99),
        ("Answer: 1234", 1234),
        ("= 55", 55),
        ("The total is -17", -17),
        (r"\boxed{1,234}", 1234),
        ("```python\nprint(42)\n```", 42),
        ("$42$", 42),
        ("Some text\n42", 42),
    ])
    def test_sum_task(self, response, expected):
        parser = UnifiedParser(task_name="sum")
        result = parser.parse(response)
        assert result.success, f"Failed to parse: {response!r}"
        assert result.value == expected, f"Expected {expected}, got {result.value!r} from {response!r}"

    # --- List task ---

    @pytest.mark.parametrize("response", [
        "[1, 2, 3, 4, 5]",
        r"\boxed{1, 2, 3, 4, 5}",
        "The sorted list is [1, 2, 3, 4, 5]",
        "1, 2, 3, 4, 5",
    ])
    def test_sorting_task(self, response):
        parser = UnifiedParser(task_name="sorting")
        result = parser.parse(response)
        assert result.success, f"Failed to parse sorted list from: {response!r}"
        import ast
        parsed = ast.literal_eval(result.value) if isinstance(result.value, str) else result.value
        assert isinstance(parsed, list), f"Expected list, got {type(result.value)}"

    # --- Comparison task ---

    @pytest.mark.parametrize("response,expected", [
        ("Number 1 is greater than Number 2", "greater than"),
        ("The result is less than", "less than"),
        (r"\boxed{greater than}", "greater than"),
        ("They are equal to each other", "equal to"),
        ("42 > 37", "greater than"),
    ])
    def test_comparison_task(self, response, expected):
        parser = UnifiedParser(task_name="comparison")
        result = parser.parse(response)
        assert result.success, f"Failed to parse comparison from: {response!r}"
        assert result.value == expected, f"Expected {expected!r}, got {result.value!r}"

    # --- Sequence task ---

    @pytest.mark.parametrize("response,expected", [
        ("The next term is 144", 144),
        (r"\boxed{89}", 89),
        ("The following term is 21", 21),
    ])
    def test_fibonacci_task(self, response, expected):
        parser = UnifiedParser(task_name="fibonacci_sequence")
        result = parser.parse(response)
        assert result.success, f"Failed to parse sequence from: {response!r}"
        assert result.value == expected

    # --- Model adapter integration ---

    def test_qwen_token_stripping(self):
        parser = UnifiedParser(task_name="sum")
        response = r"The answer is \boxed{42}<|im_end|>"
        result = parser.parse(response, model_id="Qwen/Qwen2.5-1.5B-Instruct")
        assert result.success
        assert result.value == 42

    def test_llama_inst_stripping(self):
        parser = UnifiedParser(task_name="sum")
        response = r"[INST] ignored [/INST] The answer is \boxed{42}"
        result = parser.parse(response, model_id="meta-llama/Llama-3.2-3B")
        assert result.success
        assert result.value == 42

    # --- Empty / edge cases ---

    def test_empty_response(self):
        parser = UnifiedParser(task_name="sum")
        result = parser.parse("")
        assert not result.success

    def test_none_response(self):
        parser = UnifiedParser(task_name="sum")
        result = parser.parse(None)
        assert not result.success

    def test_garbage_response(self):
        parser = UnifiedParser(task_name="sum")
        result = parser.parse("aaabbbccc!!!###$$$")
        # Should not crash; may or may not find a number
        assert isinstance(result, ParseResult)

    def test_large_response(self):
        parser = UnifiedParser(task_name="sum")
        response = "thinking... " * 500 + r"\boxed{42}"
        result = parser.parse(response)
        assert result.success
        assert result.value == 42

    # --- Type conversion ---

    def test_int_conversion(self):
        parser = UnifiedParser(task_name="sum")
        result = parser.parse(r"\boxed{42}")
        assert isinstance(result.value, int)

    def test_float_conversion(self):
        parser = UnifiedParser(task_name="division")
        result = parser.parse(r"\boxed{3.14}")
        assert isinstance(result.value, float)

    def test_list_conversion(self):
        parser = UnifiedParser(task_name="sorting")
        result = parser.parse("[1, 2, 3, 4, 5]")
        assert result.success
        assert isinstance(result.value, list)

    # --- parse_response convenience function ---

    def test_parse_response_function(self):
        result = parse_response(r"\boxed{42}", task_name="sum")
        assert result.success
        assert result.value == 42

    # --- Legacy fallback ---

    def test_legacy_fallback_used_on_failure(self):
        def legacy_fn(response):
            return True, 42

        parser = UnifiedParser(task_name="sum")
        instr, answer = parser.parse_with_legacy_fallback(
            "no numbers here at all!!!",
            legacy_fn=legacy_fn,
            log_disagreement=False,
        )
        assert answer == 42

    def test_unified_preferred_over_legacy(self):
        def legacy_fn(response):
            return False, 99  # Legacy would return 99

        parser = UnifiedParser(task_name="sum")
        instr, answer = parser.parse_with_legacy_fallback(
            r"\boxed{42}",  # Unified should find 42
            legacy_fn=legacy_fn,
            log_disagreement=False,
        )
        # Unified parser wins
        assert answer == 42


# ============================================================================
# Regression tests — parse success rate
# ============================================================================

REGRESSION_SAMPLES = [
    # (task_name, response, expected_value_or_none)
    ("sum", r"\boxed{42}", 42),
    ("sum", "Answer: 100", 100),
    ("sum", "The sum of the numbers is 1234", 1234),
    ("sum", "= -17", -17),
    ("sum", r"$$\boxed{99}$$", 99),
    ("sum", "```python\nprint(55)\n```", 55),
    ("sum", "The answer is approximately 42", 42),
    ("sum", "42 is the answer", 42),
    ("sorting", "[1, 2, 3, 4, 5]", [1, 2, 3, 4, 5]),
    ("sorting", r"\boxed{1, 2, 3, 4, 5}", [1, 2, 3, 4, 5]),
    ("comparison", "greater than", "greater than"),
    ("comparison", "less than", "less than"),
    ("comparison", "equal to", "equal to"),
    ("comparison", r"\boxed{greater than}", "greater than"),
    ("fibonacci_sequence", "The next term is 144", 144),
    ("fibonacci_sequence", r"\boxed{89}", 89),
    ("division", r"\boxed{3.5}", 3.5),
    ("even_count", r"\boxed{3}", 3),
    ("mean", r"\boxed{5.5}", 5.5),
]


@pytest.mark.parametrize("task_name,response,expected", REGRESSION_SAMPLES)
def test_regression_parse_success(task_name, response, expected):
    """Ensure all known-good samples still parse correctly."""
    parser = UnifiedParser(task_name=task_name)
    result = parser.parse(response)
    assert result.success, f"Regression: failed to parse {response!r} for task {task_name!r}"
    if isinstance(expected, list):
        import ast
        val = ast.literal_eval(result.value) if isinstance(result.value, str) else result.value
        assert val == expected, f"Expected {expected}, got {val}"
    else:
        assert result.value == expected, \
            f"Expected {expected!r}, got {result.value!r} for task={task_name!r} response={response!r}"


# ============================================================================
# Import smoke test
# ============================================================================

def test_imports():
    """Verify all Phase 1 components import cleanly."""
    from beyondbench.parsers import (
        UnifiedParser,
        parse_response,
        ParserConfig,
        get_task_config,
        TASK_PARSER_CONFIGS,
        get_adapter,
        set_parser_mode,
        get_parser_mode,
    )
    assert UnifiedParser is not None
    assert len(TASK_PARSER_CONFIGS) >= 10


def test_parser_mode_toggle():
    from beyondbench.parsers.settings import set_parser_mode, get_parser_mode
    set_parser_mode("legacy")
    assert get_parser_mode() == "legacy"
    set_parser_mode("unified")
    assert get_parser_mode() == "unified"


def test_parser_mode_invalid():
    from beyondbench.parsers.settings import set_parser_mode
    with pytest.raises(ValueError):
        set_parser_mode("invalid_mode")
