"""
Per-task parser configurations for UnifiedParser.

Each entry in TASK_PARSER_CONFIGS maps a task name (matching task_name property)
to a ParserConfig specifying expected_type, priority strategies, tolerance,
and optional post-processors.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any


@dataclass
class ParserConfig:
    """
    Configuration for how UnifiedParser should handle a specific task.

    Attributes:
        expected_type: One of "int", "float", "str", "list", "grid", "comparison", "boolean".
        strategies: Ordered list of strategy names to try (highest priority first).
        tolerance: Numeric tolerance for float comparisons (default 0 = exact match).
        post_processors: Optional list of callables to transform the extracted value.
        description: Human-readable description for debugging.
    """
    expected_type: str
    strategies: List[str]
    tolerance: float = 0.0
    post_processors: List[Callable] = field(default_factory=list)
    description: str = ""


# ============================================================================
# Easy Suite Tasks (29 tasks)
# ============================================================================

TASK_PARSER_CONFIGS: dict[str, ParserConfig] = {

    # --- Numeric tasks (single integer/float answer) ---
    "sum": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Sum of a list of integers",
    ),
    "subtraction": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Subtraction result",
    ),
    "multiplication": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Multiplication result",
    ),
    "division": ParserConfig(
        expected_type="float",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-6,
        description="Division result (float)",
    ),
    "absolute_difference": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Absolute difference of two integers",
    ),
    "mean": ParserConfig(
        expected_type="float",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-6,
        description="Arithmetic mean",
    ),
    "median": ParserConfig(
        expected_type="float",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-6,
        description="Median value",
    ),
    "mode": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Mode of a list",
    ),
    "find_maximum": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Maximum element",
    ),
    "find_minimum": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Minimum element",
    ),
    "second_maximum": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Second maximum element",
    ),
    "range": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Range (max - min)",
    ),
    "index_of_maximum": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Index of maximum element (0-based)",
    ),
    "max_adjacent_difference": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Maximum adjacent difference",
    ),
    "alternating_sum": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Alternating sum (+/-) of a list",
    ),
    "sum_of_digits": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Sum of digits of a number",
    ),
    "sum_of_max_indices": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Sum at max indices",
    ),
    "longest_increasing_subsequence": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Length of longest increasing subsequence",
    ),

    # --- Count tasks (integer) ---
    "even_count": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of even numbers",
    ),
    "odd_count": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of odd numbers",
    ),
    "count_negative": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of negative numbers",
    ),
    "count_unique": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of unique elements",
    ),
    "count_greater_than_previous": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of elements greater than previous",
    ),
    "count_palindromic": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of palindromic numbers",
    ),
    "count_perfect_squares": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of perfect squares",
    ),
    "count_multiples": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of multiples of a given number",
    ),
    "local_maxima_count": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Count of local maxima in a list",
    ),

    # ============================================================================
    # Phase 12: 15 new easy tasks
    # ============================================================================

    # --- Scalar int/float tasks ---
    "weighted_sum": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Weighted sum: sum of i * a[i-1] for 1-indexed positions",
    ),
    "second_minimum": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Second smallest distinct element",
    ),
    "dot_product": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Dot product of two equal-length lists",
    ),
    "variance": ParserConfig(
        expected_type="float",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-2,
        description="Population variance",
    ),
    "standard_deviation": ParserConfig(
        expected_type="float",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-2,
        description="Population standard deviation",
    ),

    # --- String tasks ---
    "parity_check": ParserConfig(
        expected_type="str",
        strategies=["boxed", "explicit_statement", "fallback"],
        tolerance=0,
        description="Product parity: 'even' or 'odd'",
    ),

    # --- List tasks ---
    "running_average": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=1e-4,
        description="Running (cumulative) average list",
    ),
    "cumulative_sum": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=0,
        description="Cumulative (prefix) sum list",
    ),
    "reverse_list": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=0,
        description="Reversed list",
    ),
    "rotate_list": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=0,
        description="Right-rotated list by k positions",
    ),
    "interleave_lists": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=0,
        description="Interleaved two lists element by element",
    ),
    "set_intersection": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=0,
        description="Sorted set intersection of two lists",
    ),
    "set_difference": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=0,
        description="Sorted set difference: elements in list1 not in list2",
    ),
    "moving_average": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=1e-4,
        description="k-window moving average list",
    ),
    "element_frequency": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=0,
        description="Frequency table [[val, count], ...] sorted by val",
    ),

    # --- List task ---
    "sorting": ParserConfig(
        expected_type="list",
        strategies=["boxed", "list", "explicit_statement", "code_block", "json", "fallback"],
        tolerance=0,
        description="Sort a list of integers",
    ),

    # --- Comparison task ---
    "comparison": ParserConfig(
        expected_type="comparison",
        strategies=["boxed", "comparison", "explicit_statement", "fallback"],
        tolerance=0,
        description="Compare two numbers (greater/less/equal)",
    ),

    # ============================================================================
    # Medium Suite Tasks (5 tasks)
    # ============================================================================

    "fibonacci_sequence": ParserConfig(
        expected_type="int",
        strategies=["boxed", "sequence", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Next term in Fibonacci sequence",
    ),
    "geometric_sequence": ParserConfig(
        expected_type="float",
        strategies=["boxed", "sequence", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-6,
        description="Next term in geometric sequence",
    ),
    "algebraic_sequence": ParserConfig(
        expected_type="float",
        strategies=["boxed", "sequence", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-6,
        description="Next term in algebraic sequence",
    ),
    "prime_sequence": ParserConfig(
        expected_type="int",
        strategies=["boxed", "sequence", "explicit_statement", "code_block", "latex_math", "fallback"],
        tolerance=0,
        description="Next prime in sequence",
    ),
    "complex_pattern": ParserConfig(
        expected_type="float",
        strategies=["boxed", "sequence", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-6,
        description="Next term in complex pattern sequence",
    ),

    # ============================================================================
    # Hard Suite Tasks (10 tasks)
    # ============================================================================

    "sudoku_solving": ParserConfig(
        expected_type="grid",
        strategies=["grid", "json", "code_block", "fallback"],
        tolerance=0,
        description="9x9 Sudoku grid solution",
    ),
    "n_queens": ParserConfig(
        expected_type="grid",
        strategies=["grid", "json", "list", "code_block", "fallback"],
        tolerance=0,
        description="N-Queens placement grid",
    ),
    "tower_hanoi": ParserConfig(
        expected_type="list",
        strategies=["explicit_statement", "list", "code_block", "json", "fallback"],
        tolerance=0,
        description="Tower of Hanoi move sequence",
    ),
    "boolean_sat": ParserConfig(
        expected_type="str",
        strategies=["boxed", "json", "explicit_statement", "code_block", "fallback"],
        tolerance=0,
        description="Boolean SAT variable assignment",
    ),
    "graph_coloring": ParserConfig(
        expected_type="str",
        strategies=["boxed", "json", "explicit_statement", "code_block", "fallback"],
        tolerance=0,
        description="Graph coloring assignment",
    ),
    "cryptarithmetic": ParserConfig(
        expected_type="int",
        strategies=["boxed", "json", "explicit_statement", "code_block", "fallback"],
        tolerance=0,
        description="Cryptarithmetic puzzle solution",
    ),
    "constraint_optimization": ParserConfig(
        expected_type="float",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=1e-6,
        description="Constraint optimization optimal value",
    ),
    "matrix_chain_multiplication": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=0,
        description="Matrix chain multiplication min operations",
    ),
    "logic_grid_puzzles": ParserConfig(
        expected_type="str",
        strategies=["boxed", "json", "explicit_statement", "code_block", "fallback"],
        tolerance=0,
        description="Logic grid puzzle solution",
    ),
    "modular_systems_solver": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=0,
        description="Modular arithmetic system solution",
    ),
    # Alias: task_registry uses "modular_systems" as the canonical key
    "modular_systems": ParserConfig(
        expected_type="int",
        strategies=["boxed", "explicit_statement", "latex_math", "code_block", "fallback"],
        tolerance=0,
        description="Modular arithmetic system solution (alias for modular_systems_solver)",
    ),
    # Alias: tower_of_hanoi was the old key; keep for backward compatibility
    "tower_of_hanoi": ParserConfig(
        expected_type="list",
        strategies=["explicit_statement", "list", "code_block", "json", "fallback"],
        tolerance=0,
        description="Tower of Hanoi move sequence (alias for tower_hanoi)",
    ),
}


def get_task_config(task_name: str) -> ParserConfig:
    """
    Return the ParserConfig for the given task name.

    Falls back to a generic numeric config if task not found.

    Args:
        task_name: The task name (matching BaseTask.task_name property).

    Returns:
        ParserConfig for the task.
    """
    if task_name in TASK_PARSER_CONFIGS:
        return TASK_PARSER_CONFIGS[task_name]

    # Generic fallback for unknown tasks
    return ParserConfig(
        expected_type="auto",
        strategies=["boxed", "explicit_statement", "code_block", "latex_math", "json", "fallback"],
        tolerance=1e-9,
        description=f"Generic fallback config for unknown task: {task_name}",
    )
