"""Test utility modules."""

import pytest


class TestSharedUtils:
    def test_values_are_close_same(self):
        from beyondbench.utils.shared_utils import values_are_close
        assert values_are_close(1.0, 1.0000001) is True

    def test_values_are_close_different(self):
        from beyondbench.utils.shared_utils import values_are_close
        assert values_are_close(1.0, 2.0) is False

    def test_values_are_close_both_none(self):
        from beyondbench.utils.shared_utils import values_are_close
        assert values_are_close(None, None) is True

    def test_values_are_close_one_none(self):
        from beyondbench.utils.shared_utils import values_are_close
        assert values_are_close(None, 1.0) is False

    def test_round_if_close_to_int_rounds(self):
        from beyondbench.utils.shared_utils import round_if_close_to_int
        assert round_if_close_to_int(3.00000000001) == 3

    def test_round_if_close_to_int_preserves_float(self):
        from beyondbench.utils.shared_utils import round_if_close_to_int
        assert round_if_close_to_int(3.5) == 3.5

    def test_round_if_close_to_int_preserves_int(self):
        from beyondbench.utils.shared_utils import round_if_close_to_int
        assert round_if_close_to_int(42) == 42

    def test_is_valid_number_int(self):
        from beyondbench.utils.shared_utils import is_valid_number
        assert is_valid_number(42) is True

    def test_is_valid_number_float(self):
        from beyondbench.utils.shared_utils import is_valid_number
        assert is_valid_number(3.14) is True

    def test_is_valid_number_string(self):
        from beyondbench.utils.shared_utils import is_valid_number
        assert is_valid_number("42") is True

    def test_is_valid_number_none(self):
        from beyondbench.utils.shared_utils import is_valid_number
        assert is_valid_number(None) is False

    def test_is_valid_number_text(self):
        from beyondbench.utils.shared_utils import is_valid_number
        assert is_valid_number("abc") is False

    def test_safe_divide_normal(self):
        from beyondbench.utils.shared_utils import safe_divide
        assert safe_divide(10, 2) == 5.0

    def test_safe_divide_by_zero(self):
        from beyondbench.utils.shared_utils import safe_divide
        assert safe_divide(10, 0) == 0.0

    def test_safe_divide_by_zero_custom_default(self):
        from beyondbench.utils.shared_utils import safe_divide
        assert safe_divide(10, 0, default=-1.0) == -1.0

    def test_clamp_within_range(self):
        from beyondbench.utils.shared_utils import clamp
        assert clamp(5, 0, 10) == 5

    def test_clamp_below_min(self):
        from beyondbench.utils.shared_utils import clamp
        assert clamp(-1, 0, 10) == 0

    def test_clamp_above_max(self):
        from beyondbench.utils.shared_utils import clamp
        assert clamp(15, 0, 10) == 10

    def test_is_prime_true(self):
        from beyondbench.utils.shared_utils import is_prime
        assert is_prime(2) is True
        assert is_prime(7) is True
        assert is_prime(13) is True

    def test_is_prime_false(self):
        from beyondbench.utils.shared_utils import is_prime
        assert is_prime(4) is False
        assert is_prime(1) is False
        assert is_prime(0) is False

    def test_fibonacci(self):
        from beyondbench.utils.shared_utils import fibonacci
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1
        assert fibonacci(10) == 55

    def test_factorial_normal(self):
        from beyondbench.utils.shared_utils import factorial
        assert factorial(0) == 1
        assert factorial(5) == 120

    def test_factorial_negative_raises(self):
        from beyondbench.utils.shared_utils import factorial
        with pytest.raises(ValueError):
            factorial(-1)

    def test_gcd(self):
        from beyondbench.utils.shared_utils import gcd
        assert gcd(12, 8) == 4
        assert gcd(7, 3) == 1

    def test_lcm(self):
        from beyondbench.utils.shared_utils import lcm
        assert lcm(4, 6) == 12
        assert lcm(3, 5) == 15


class TestErrorHandler:
    def test_rate_limit_detection_keywords(self):
        from beyondbench.utils.error_handler import ErrorHandler
        handler = ErrorHandler()
        assert handler.is_rate_limit_error("rate limit exceeded") is True
        assert handler.is_rate_limit_error("429 Too Many Requests") is True

    def test_rate_limit_detection_negative(self):
        from beyondbench.utils.error_handler import ErrorHandler
        handler = ErrorHandler()
        assert handler.is_rate_limit_error("normal error") is False

    def test_memory_error_detection(self):
        from beyondbench.utils.error_handler import ErrorHandler
        handler = ErrorHandler()
        assert handler.is_memory_error("CUDA out of memory") is True
        assert handler.is_memory_error("normal error") is False

    def test_timeout_error_detection(self):
        from beyondbench.utils.error_handler import ErrorHandler
        handler = ErrorHandler()
        assert handler.is_timeout_error("connection timeout") is True
        assert handler.is_timeout_error("normal error") is False

    def test_error_category_rate_limit(self):
        from beyondbench.utils.error_handler import ErrorHandler
        handler = ErrorHandler()
        assert handler.get_error_category("rate limit") == "rate_limit"

    def test_error_category_memory(self):
        from beyondbench.utils.error_handler import ErrorHandler
        handler = ErrorHandler()
        assert handler.get_error_category("out of memory") == "memory"

    def test_error_category_timeout(self):
        from beyondbench.utils.error_handler import ErrorHandler
        handler = ErrorHandler()
        assert handler.get_error_category("timeout") == "timeout"

    def test_error_category_unknown(self):
        from beyondbench.utils.error_handler import ErrorHandler
        handler = ErrorHandler()
        assert handler.get_error_category("something else") == "unknown"


class TestStatsTracker:
    def test_init(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        tracker = StatsTracker()
        assert tracker is not None

    def test_update_and_summary(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        tracker = StatsTracker()
        tracker.update_stats({"task_name": "sum", "status": "success", "evaluation": {"accuracy": 0.9}})
        tracker.update_stats({"task_name": "sum", "status": "success", "evaluation": {"accuracy": 0.8}})
        summary = tracker.get_summary()
        assert "sum" in summary
        assert summary["sum"]["total_evaluations"] == 2

    def test_summary_has_overall(self):
        from beyondbench.utils.stats_tracker import StatsTracker
        tracker = StatsTracker()
        tracker.update_stats({"task_name": "sum", "status": "success", "evaluation": {"accuracy": 0.9}})
        summary = tracker.get_summary()
        assert "overall" in summary
        assert summary["overall"]["total_tasks"] == 1


class TestTokenCounter:
    def test_init(self):
        from beyondbench.utils.token_counter import TokenCounter
        counter = TokenCounter()
        assert counter is not None

    def test_count_tokens(self):
        from beyondbench.utils.token_counter import TokenCounter
        counter = TokenCounter()
        count = counter.count_tokens("Hello world, this is a test.")
        assert count > 0

    def test_count_empty(self):
        from beyondbench.utils.token_counter import TokenCounter
        counter = TokenCounter()
        assert counter.count_tokens("") == 0


class TestReportGenerator:
    def test_efficiency_score(self):
        from beyondbench.utils.report_generator import calculate_efficiency_score
        score = calculate_efficiency_score(0.9, 50, [30, 50, 100])
        assert 0 <= score <= 1

    def test_efficiency_score_single_model(self):
        from beyondbench.utils.report_generator import calculate_efficiency_score
        score = calculate_efficiency_score(0.9, 50, [50])
        assert score == 0.9

    def test_overthinking_ratio(self):
        from beyondbench.utils.report_generator import calculate_overthinking_ratio
        ratio = calculate_overthinking_ratio(0.9, 100, 50)
        assert ratio > 0

    def test_overthinking_ratio_zero_baseline(self):
        from beyondbench.utils.report_generator import calculate_overthinking_ratio
        ratio = calculate_overthinking_ratio(0.9, 100, 0)
        assert ratio == 1.0
