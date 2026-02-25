"""
Task-specific parsers for beyondbench package.

These parsers provide robust, task-specific answer extraction for all easy suite tasks.
They use the unified parsing utilities from beyondbench.utils.parsing as their foundation.
"""

# Import unified parsing functions
from ..utils.parsing import (
    parse_boxed_answer,
    parse_number,
    parse_number_list,
    parse_boolean,
    parse_sorted_list,
    parse_count,
    parse_sum,
    parse_index,
    parse_comparison_result,
    parse_arithmetic_result,
    parse_statistical_result,
    parse_sequence_result,
    parse_boolean_result
)

# Import individual task parsers for backward compatibility
from .absolute_difference_parsing import parse_absolute_difference_answer
from .comparison_parsing import parse_comparison_result as parse_comparison_result_legacy
from .division_parsing import parse_division_answer
from .even_count_parsing import parse_even_count_answer
from .find_maximum_parsing import parse_find_maximum_answer
from .find_minimum_parsing import parse_find_minimum_answer
from .multiplication_parsing import parse_multiplication_answer
from .odd_count_parsing import parse_odd_count_answer
from .sorting_parsing import parse_sorted_list as parse_sorted_list_legacy
from .sum_parsing import parse_sum_answer
from .mean_parsing import parse_mean_answer
from .median_parsing import parse_median_answer
from .mode_parsing import parse_mode_answer
from .subtraction_parsing import parse_subtraction_answer

__all__ = [
    # Unified parsing functions (preferred)
    "parse_boxed_answer",
    "parse_number",
    "parse_number_list",
    "parse_boolean",
    "parse_sorted_list",
    "parse_count",
    "parse_sum",
    "parse_index",
    "parse_comparison_result",
    "parse_arithmetic_result",
    "parse_statistical_result",
    "parse_sequence_result",
    "parse_boolean_result",

    # Legacy task-specific parsers (for backward compatibility)
    "parse_absolute_difference_answer",
    "parse_comparison_result_legacy",
    "parse_division_answer",
    "parse_even_count_answer",
    "parse_find_maximum_answer",
    "parse_find_minimum_answer",
    "parse_multiplication_answer",
    "parse_odd_count_answer",
    "parse_sorted_list_legacy",
    "parse_sum_answer",
    "parse_mean_answer",
    "parse_median_answer",
    "parse_mode_answer",
    "parse_subtraction_answer"
]