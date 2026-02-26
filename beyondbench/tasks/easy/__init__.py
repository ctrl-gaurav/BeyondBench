"""Easy suite task implementations."""

# Import all task classes from the easy suite
try:
    from .sorting_task import SortingTask
    from .comparison_task import ComparisonTask
    from .sum_task import SumTask
    from .multiplication_task import MultiplicationTask
    from .odd_count_task import OddCountTask
    from .even_count_task import EvenCountTask
    from .absolute_difference_task import AbsoluteDifferenceTask
    from .division_task import DivisionTask
    from .find_maximum_task import FindMaximumTask
    from .find_minimum_task import FindMinimumTask
    from .mean_task import MeanTask
    from .median_task import MedianTask
    from .mode_task import ModeTask
    from .subtraction_task import SubtractionTask

    # New tasks from the 15 additional scalable tasks
    from .second_maximum_task import SecondMaximumTask
    from .range_task import RangeTask
    from .index_of_maximum_task import IndexOfMaximumTask
    from .count_negative_task import CountNegativeTask
    from .count_unique_task import CountUniqueTask
    from .max_adjacent_difference_task import MaxAdjacentDifferenceTask
    from .count_greater_than_previous_task import CountGreaterThanPreviousTask
    from .sum_of_max_indices_task import SumOfMaxIndicesTask
    from .count_palindromic_task import CountPalindromicTask
    from .longest_increasing_subsequence_task import LongestIncreasingSubsequenceTask
    from .sum_of_digits_task import SumOfDigitsTask
    from .count_perfect_squares_task import CountPerfectSquaresTask
    from .alternating_sum_task import AlternatingSumTask
    from .count_multiples_task import CountMultiplesTask
    from .local_maxima_count_task import LocalMaximaCountTask

except ImportError as e:
    import logging
    logging.debug(f"Failed to import some easy task classes: {e}")

__all__ = [
    # Original tasks
    "SortingTask", "ComparisonTask", "SumTask", "MultiplicationTask",
    "OddCountTask", "EvenCountTask", "AbsoluteDifferenceTask", "DivisionTask",
    "FindMaximumTask", "FindMinimumTask", "MeanTask", "MedianTask",
    "ModeTask", "SubtractionTask",

    # New scalable tasks
    "SecondMaximumTask", "RangeTask", "IndexOfMaximumTask", "CountNegativeTask",
    "CountUniqueTask", "MaxAdjacentDifferenceTask", "CountGreaterThanPreviousTask",
    "SumOfMaxIndicesTask", "CountPalindromicTask", "LongestIncreasingSubsequenceTask",
    "SumOfDigitsTask", "CountPerfectSquaresTask", "AlternatingSumTask",
    "CountMultiplesTask", "LocalMaximaCountTask"
]