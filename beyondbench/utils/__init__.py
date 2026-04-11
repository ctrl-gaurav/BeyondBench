"""
Utility components for beyondbench package.

This module provides essential utilities for the beyondbench package including
parsing, logging, error handling, and other common functionality.
"""

try:
    # Core utilities
    from .parsing import (
        parse_boxed_answer,
        parse_sorted_list,
        parse_count,
        parse_sum,
        parse_index,
        parse_comparison_result,
        parse_arithmetic_result,
        parse_statistical_result,
        parse_sequence_result,
        parse_boolean_result,
        validate_sorted_list,
        validate_count_result,
        validate_arithmetic_result,
        validate_index_result
    )

    from .logging_utils import get_logger, setup_logging
    from .error_handler import ErrorHandler
    from .token_counter import TokenCounter, TokenUsage, TokenAnalytics
    from .stats_tracker import StatsTracker
    from .report_generator import ReportGenerator, generate_final_report
    from .cost_tracker import CostTracker, CostReport, TaskCost
    from .request_logger import RequestLogger, RequestEntry
    from .visualizer import Visualizer
    from .shared_utils import (
        values_are_close,
        round_if_close_to_int,
        is_valid_number,
        safe_divide,
        clamp,
        calculate_percentage,
        format_number,
        is_power_of_two,
        gcd,
        lcm,
        fibonacci,
        is_prime,
        factorial
    )
    from .sequence_parsing_utils import SequenceAnswerParser, parse_sequence_answer

except ImportError as e:
    print(f"Warning: Failed to import some utility components: {e}")

__all__ = [
    # Parsing utilities (main exports)
    "parse_boxed_answer",
    "parse_sorted_list",
    "parse_count",
    "parse_sum",
    "parse_index",
    "parse_comparison_result",
    "parse_arithmetic_result",
    "parse_statistical_result",
    "parse_sequence_result",
    "parse_boolean_result",
    "validate_sorted_list",
    "validate_count_result",
    "validate_arithmetic_result",
    "validate_index_result",

    # Sequence parsing
    "SequenceAnswerParser",
    "parse_sequence_answer",

    # Shared utilities
    "values_are_close",
    "round_if_close_to_int",
    "is_valid_number",
    "safe_divide",
    "clamp",
    "calculate_percentage",
    "format_number",
    "is_power_of_two",
    "gcd",
    "lcm",
    "fibonacci",
    "is_prime",
    "factorial",

    # Other utilities
    "get_logger",
    "setup_logging",
    "ErrorHandler",
    "TokenCounter",
    "TokenUsage",
    "TokenAnalytics",
    "StatsTracker",
    "ReportGenerator",
    "generate_final_report",
    "CostTracker",
    "CostReport",
    "TaskCost",
    "RequestLogger",
    "RequestEntry",
    "Visualizer",
]