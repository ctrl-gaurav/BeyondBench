"""Register concise / detailed / few_shot prompt variants for all 79 BeyondBench tasks.

This module is imported automatically by PromptLibrary.__init__() via a side-effect
import.  Every call to ``_reg(task_name, style, template_str)`` registers one
PromptTemplate.

IMPORTANT
---------
* All prompts MUST end with a ``\\boxed{answer}`` instruction — parsers rely on it.
* The *data* placeholder receives ``str(data_point)`` from BaseTask.get_prompt().
  For complex/dict data-points the value may already be a formatted string.
* few_shot examples use tiny, hand-crafted instances to keep token counts low.
"""

from __future__ import annotations

from .template import PromptTemplate

# _registry is injected by PromptLibrary.__init__ before this module is imported.
# We use a module-level dict reference so we never need to acquire the class lock
# (which would deadlock since __init__ holds it while importing this module).
_registry: dict = {}


def _reg(task_name: str, style: str, template_str: str, description: str = "") -> None:
    name = f"{task_name}_{style}"
    tmpl = PromptTemplate(name=name, template_str=template_str, style=style, description=description)
    if task_name not in _registry:
        _registry[task_name] = {}
    _registry[task_name][style] = tmpl


# ===========================================================================
# EASY TASKS
# ===========================================================================

# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------
_reg("sum", "concise",
     "Sum these numbers: {data}\nAnswer in \\boxed{{answer}} format.")

_reg("sum", "detailed",
     "You are given a list of numbers.\n\n"
     "List: {data}\n\n"
     "Calculate the sum of all numbers in the list. "
     "Add each number sequentially. "
     "Present your final numerical answer as \\boxed{{answer}}.")

_reg("sum", "few_shot",
     "Example 1: [1, 2, 3] → sum = 6 → \\boxed{{6}}\n"
     "Example 2: [-1, 4, 2] → sum = 5 → \\boxed{{5}}\n\n"
     "Now compute the sum of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# sorting
# ---------------------------------------------------------------------------
_reg("sorting", "concise",
     "Sort this list in ascending order: {data}\n"
     "Provide the sorted list as \\boxed{{answer}}.")

_reg("sorting", "detailed",
     "You are given the following list of numbers:\n\n{data}\n\n"
     "Rearrange all elements in ascending (smallest to largest) order. "
     "Show your sorted list. "
     "Write your final sorted list as \\boxed{{answer}}.")

_reg("sorting", "few_shot",
     "Example 1: [3, 1, 2] → sorted: [1, 2, 3] → \\boxed{{[1, 2, 3]}}\n"
     "Example 2: [5, -1, 0] → sorted: [-1, 0, 5] → \\boxed{{[-1, 0, 5]}}\n\n"
     "Now sort: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
_reg("comparison", "concise",
     "Compare these two numbers: {data}\n"
     "Answer 'greater', 'less', or 'equal' as \\boxed{{answer}}.")

_reg("comparison", "detailed",
     "Given the following two numbers:\n\n{data}\n\n"
     "Determine whether the first number is greater than, less than, or equal to the second number. "
     "Respond with exactly one word: 'greater', 'less', or 'equal'. "
     "Write your answer as \\boxed{{answer}}.")

_reg("comparison", "few_shot",
     "Example 1: compare 5 and 3 → 5 > 3 → \\boxed{{greater}}\n"
     "Example 2: compare 2 and 7 → 2 < 7 → \\boxed{{less}}\n"
     "Example 3: compare 4 and 4 → equal → \\boxed{{equal}}\n\n"
     "Now evaluate: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# multiplication
# ---------------------------------------------------------------------------
_reg("multiplication", "concise",
     "Multiply these numbers: {data}\nAnswer as \\boxed{{answer}}.")

_reg("multiplication", "detailed",
     "You are given a list of numbers:\n\n{data}\n\n"
     "Compute the product of all numbers in the list (multiply them all together). "
     "Show your work if helpful. "
     "Write the final product as \\boxed{{answer}}.")

_reg("multiplication", "few_shot",
     "Example 1: [2, 3, 4] → 2 × 3 × 4 = 24 → \\boxed{{24}}\n"
     "Example 2: [-1, 5, 2] → -1 × 5 × 2 = -10 → \\boxed{{-10}}\n\n"
     "Now multiply: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# odd_count
# ---------------------------------------------------------------------------
_reg("odd_count", "concise",
     "Count the odd numbers in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("odd_count", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Count how many numbers are odd (not divisible by 2). "
     "A number n is odd when n % 2 ≠ 0. "
     "Write the count as \\boxed{{answer}}.")

_reg("odd_count", "few_shot",
     "Example 1: [1, 2, 3, 4] → odd numbers: 1, 3 → count = 2 → \\boxed{{2}}\n"
     "Example 2: [2, 4, 6] → no odd numbers → count = 0 → \\boxed{{0}}\n\n"
     "Now count odds in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# even_count
# ---------------------------------------------------------------------------
_reg("even_count", "concise",
     "Count the even numbers in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("even_count", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Count how many numbers are even (divisible by 2, including 0). "
     "Write the count as \\boxed{{answer}}.")

_reg("even_count", "few_shot",
     "Example 1: [1, 2, 3, 4] → even numbers: 2, 4 → count = 2 → \\boxed{{2}}\n"
     "Example 2: [1, 3, 5] → no even numbers → count = 0 → \\boxed{{0}}\n\n"
     "Now count evens in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# absolute_difference
# ---------------------------------------------------------------------------
_reg("absolute_difference", "concise",
     "Find the absolute difference between the two numbers: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("absolute_difference", "detailed",
     "Given the two numbers:\n\n{data}\n\n"
     "Compute the absolute difference: |a - b|. "
     "Subtract one from the other and take the absolute value. "
     "Write the result as \\boxed{{answer}}.")

_reg("absolute_difference", "few_shot",
     "Example 1: 7 and 3 → |7 - 3| = 4 → \\boxed{{4}}\n"
     "Example 2: -2 and 5 → |-2 - 5| = 7 → \\boxed{{7}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# division
# ---------------------------------------------------------------------------
_reg("division", "concise",
     "Divide the first number by the second: {data}\n"
     "Round to 2 decimal places if needed. Answer as \\boxed{{answer}}.")

_reg("division", "detailed",
     "You are given two numbers:\n\n{data}\n\n"
     "Divide the first number by the second number. "
     "If the result is not an integer, round to 2 decimal places. "
     "Write the final quotient as \\boxed{{answer}}.")

_reg("division", "few_shot",
     "Example 1: 10 ÷ 2 = 5 → \\boxed{{5}}\n"
     "Example 2: 7 ÷ 2 = 3.5 → \\boxed{{3.5}}\n\n"
     "Now divide: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# find_maximum
# ---------------------------------------------------------------------------
_reg("find_maximum", "concise",
     "Find the maximum value in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("find_maximum", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the largest (maximum) value in the list. "
     "Scan through all elements and identify the greatest one. "
     "Write it as \\boxed{{answer}}.")

_reg("find_maximum", "few_shot",
     "Example 1: [3, 7, 1, 5] → max = 7 → \\boxed{{7}}\n"
     "Example 2: [-5, -1, -3] → max = -1 → \\boxed{{-1}}\n\n"
     "Now find the max of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# find_minimum
# ---------------------------------------------------------------------------
_reg("find_minimum", "concise",
     "Find the minimum value in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("find_minimum", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the smallest (minimum) value in the list. "
     "Scan through all elements and identify the least one. "
     "Write it as \\boxed{{answer}}.")

_reg("find_minimum", "few_shot",
     "Example 1: [3, 7, 1, 5] → min = 1 → \\boxed{{1}}\n"
     "Example 2: [-5, -1, -3] → min = -5 → \\boxed{{-5}}\n\n"
     "Now find the min of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------
_reg("mean", "concise",
     "Calculate the arithmetic mean of: {data}\n"
     "Round to 2 decimal places. Answer as \\boxed{{answer}}.")

_reg("mean", "detailed",
     "Given the list of numbers:\n\n{data}\n\n"
     "Compute the arithmetic mean: divide the sum of all values by the count of values. "
     "Round to 2 decimal places if necessary. "
     "Write the mean as \\boxed{{answer}}.")

_reg("mean", "few_shot",
     "Example 1: [2, 4, 6] → sum = 12, count = 3, mean = 4.0 → \\boxed{{4.0}}\n"
     "Example 2: [1, 2, 3, 4] → sum = 10, count = 4, mean = 2.5 → \\boxed{{2.5}}\n\n"
     "Now compute the mean of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------
_reg("median", "concise",
     "Find the median of: {data}\nAnswer as \\boxed{{answer}}.")

_reg("median", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the median. First sort the list. If the count is odd, the median is the middle element. "
     "If the count is even, the median is the average of the two middle elements. "
     "Write the median as \\boxed{{answer}}.")

_reg("median", "few_shot",
     "Example 1: [1, 3, 5] → sorted: [1,3,5], median = 3 → \\boxed{{3}}\n"
     "Example 2: [2, 4, 6, 8] → sorted: [2,4,6,8], median = (4+6)/2 = 5.0 → \\boxed{{5.0}}\n\n"
     "Now find the median of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------
_reg("mode", "concise",
     "Find the mode (most frequent value) of: {data}\nAnswer as \\boxed{{answer}}.")

_reg("mode", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the mode: the value that appears most frequently. "
     "If multiple values tie, return the smallest. "
     "Write the mode as \\boxed{{answer}}.")

_reg("mode", "few_shot",
     "Example 1: [1, 2, 2, 3] → 2 appears twice → mode = 2 → \\boxed{{2}}\n"
     "Example 2: [4, 1, 4, 5] → 4 appears twice → mode = 4 → \\boxed{{4}}\n\n"
     "Now find the mode of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# subtraction
# ---------------------------------------------------------------------------
_reg("subtraction", "concise",
     "Subtract: {data}\nAnswer as \\boxed{{answer}}.")

_reg("subtraction", "detailed",
     "Given the following numbers:\n\n{data}\n\n"
     "Subtract the second number from the first (or perform the indicated subtraction). "
     "Write the result as \\boxed{{answer}}.")

_reg("subtraction", "few_shot",
     "Example 1: 10 - 3 = 7 → \\boxed{{7}}\n"
     "Example 2: -2 - 5 = -7 → \\boxed{{-7}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# second_maximum
# ---------------------------------------------------------------------------
_reg("second_maximum", "concise",
     "Find the second largest value in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("second_maximum", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the second maximum: the largest value that is strictly less than the maximum. "
     "Sort the unique values and pick the second largest. "
     "Write it as \\boxed{{answer}}.")

_reg("second_maximum", "few_shot",
     "Example 1: [1, 5, 3, 5, 2] → unique sorted: [1,2,3,5], 2nd max = 3 → \\boxed{{3}}\n"
     "Example 2: [7, 2, 9] → 2nd max = 7 → \\boxed{{7}}\n\n"
     "Now find the second maximum of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# range
# ---------------------------------------------------------------------------
_reg("range", "concise",
     "Find the range (max - min) of: {data}\nAnswer as \\boxed{{answer}}.")

_reg("range", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Compute the range: maximum value minus minimum value. "
     "Write the result as \\boxed{{answer}}.")

_reg("range", "few_shot",
     "Example 1: [2, 8, 5] → max=8, min=2, range=6 → \\boxed{{6}}\n"
     "Example 2: [-3, 1, 4] → max=4, min=-3, range=7 → \\boxed{{7}}\n\n"
     "Now find the range of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# index_of_maximum
# ---------------------------------------------------------------------------
_reg("index_of_maximum", "concise",
     "Find the index (0-based) of the maximum value in: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("index_of_maximum", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the 0-based index of the maximum (largest) element. "
     "If there are ties, return the index of the first occurrence. "
     "Write the index as \\boxed{{answer}}.")

_reg("index_of_maximum", "few_shot",
     "Example 1: [3, 7, 1] → max=7 at index 1 → \\boxed{{1}}\n"
     "Example 2: [5, 5, 2] → max=5 first at index 0 → \\boxed{{0}}\n\n"
     "Now find the index of max in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# count_negative
# ---------------------------------------------------------------------------
_reg("count_negative", "concise",
     "Count the negative numbers in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("count_negative", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Count how many numbers are strictly negative (less than 0). "
     "Write the count as \\boxed{{answer}}.")

_reg("count_negative", "few_shot",
     "Example 1: [-1, 2, -3, 0] → negatives: -1, -3 → count=2 → \\boxed{{2}}\n"
     "Example 2: [1, 2, 3] → no negatives → count=0 → \\boxed{{0}}\n\n"
     "Now count negatives in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# count_unique
# ---------------------------------------------------------------------------
_reg("count_unique", "concise",
     "Count distinct unique values in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("count_unique", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Count how many distinct (unique) values appear, ignoring duplicates. "
     "Write the count as \\boxed{{answer}}.")

_reg("count_unique", "few_shot",
     "Example 1: [1, 2, 2, 3] → unique: {1,2,3} → count=3 → \\boxed{{3}}\n"
     "Example 2: [5, 5, 5] → unique: {5} → count=1 → \\boxed{{1}}\n\n"
     "Now count unique values in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# max_adjacent_difference
# ---------------------------------------------------------------------------
_reg("max_adjacent_difference", "concise",
     "Find the maximum absolute difference between adjacent elements in: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("max_adjacent_difference", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "For each consecutive pair (a, b), compute |a - b|. "
     "Find the maximum of these differences. "
     "Write the result as \\boxed{{answer}}.")

_reg("max_adjacent_difference", "few_shot",
     "Example 1: [1, 5, 2] → |1-5|=4, |5-2|=3 → max=4 → \\boxed{{4}}\n"
     "Example 2: [3, 3, 8] → |3-3|=0, |3-8|=5 → max=5 → \\boxed{{5}}\n\n"
     "Now find max adjacent diff in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# count_greater_than_previous
# ---------------------------------------------------------------------------
_reg("count_greater_than_previous", "concise",
     "Count how many elements are greater than their immediate predecessor in: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("count_greater_than_previous", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Count the number of elements (starting from index 1) that are strictly greater than "
     "the element immediately before them. "
     "Write the count as \\boxed{{answer}}.")

_reg("count_greater_than_previous", "few_shot",
     "Example 1: [1, 3, 2, 5] → 3>1 (yes), 2>3 (no), 5>2 (yes) → count=2 → \\boxed{{2}}\n"
     "Example 2: [5, 4, 3] → no element exceeds predecessor → count=0 → \\boxed{{0}}\n\n"
     "Now count in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# sum_of_max_indices
# ---------------------------------------------------------------------------
_reg("sum_of_max_indices", "concise",
     "Find the sum of the indices of the maximum value in: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("sum_of_max_indices", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the maximum value. Sum all the (0-based) indices where this maximum value occurs. "
     "Write the result as \\boxed{{answer}}.")

_reg("sum_of_max_indices", "few_shot",
     "Example 1: [3, 7, 7] → max=7 at indices 1,2 → sum=3 → \\boxed{{3}}\n"
     "Example 2: [5, 1, 5] → max=5 at indices 0,2 → sum=2 → \\boxed{{2}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# count_palindromic
# ---------------------------------------------------------------------------
_reg("count_palindromic", "concise",
     "Count how many numbers in the list are palindromes (read the same forwards and backwards): {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("count_palindromic", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "A number is palindromic if its digit sequence reads the same forwards and backwards "
     "(e.g., 121, 33, 5). Count how many numbers in the list are palindromic. "
     "Write the count as \\boxed{{answer}}.")

_reg("count_palindromic", "few_shot",
     "Example 1: [121, 13, 22, 5] → palindromes: 121, 22, 5 → count=3 → \\boxed{{3}}\n"
     "Example 2: [12, 34] → no palindromes → count=0 → \\boxed{{0}}\n\n"
     "Now count palindromes in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# longest_increasing_subsequence
# ---------------------------------------------------------------------------
_reg("longest_increasing_subsequence", "concise",
     "Find the length of the longest strictly increasing subsequence of: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("longest_increasing_subsequence", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the length of the Longest Increasing Subsequence (LIS): "
     "the longest subsequence (not necessarily contiguous) where each element is strictly "
     "greater than the previous one. "
     "Write the length as \\boxed{{answer}}.")

_reg("longest_increasing_subsequence", "few_shot",
     "Example 1: [3, 1, 4, 2] → LIS is [1, 2] or [1, 4], length=2 → \\boxed{{2}}\n"
     "Example 2: [1, 2, 3] → LIS is [1,2,3], length=3 → \\boxed{{3}}\n\n"
     "Now find LIS length of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# sum_of_digits
# ---------------------------------------------------------------------------
_reg("sum_of_digits", "concise",
     "Find the sum of all digits of all numbers in: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("sum_of_digits", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "For each number, extract all its digits and sum them together across all numbers in the list. "
     "Ignore negative signs (use absolute values). "
     "Write the total digit sum as \\boxed{{answer}}.")

_reg("sum_of_digits", "few_shot",
     "Example 1: [12, 34] → digits: 1+2+3+4 = 10 → \\boxed{{10}}\n"
     "Example 2: [100, 5] → digits: 1+0+0+5 = 6 → \\boxed{{6}}\n\n"
     "Now sum digits of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# count_perfect_squares
# ---------------------------------------------------------------------------
_reg("count_perfect_squares", "concise",
     "Count perfect squares in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("count_perfect_squares", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Count how many numbers are perfect squares (i.e., equal to n² for some non-negative integer n). "
     "Examples: 0, 1, 4, 9, 16, 25, 36, ... "
     "Write the count as \\boxed{{answer}}.")

_reg("count_perfect_squares", "few_shot",
     "Example 1: [1, 3, 4, 9] → perfect squares: 1, 4, 9 → count=3 → \\boxed{{3}}\n"
     "Example 2: [2, 5, 7] → no perfect squares → count=0 → \\boxed{{0}}\n\n"
     "Now count perfect squares in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# alternating_sum
# ---------------------------------------------------------------------------
_reg("alternating_sum", "concise",
     "Compute the alternating sum (add evenly-indexed, subtract oddly-indexed elements) of: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("alternating_sum", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Compute the alternating sum: starting from index 0, add elements at even indices "
     "and subtract elements at odd indices (a[0] - a[1] + a[2] - a[3] + ...). "
     "Write the result as \\boxed{{answer}}.")

_reg("alternating_sum", "few_shot",
     "Example 1: [1, 2, 3] → 1 - 2 + 3 = 2 → \\boxed{{2}}\n"
     "Example 2: [5, 1, 4, 2] → 5 - 1 + 4 - 2 = 6 → \\boxed{{6}}\n\n"
     "Now compute alternating sum of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# count_multiples
# ---------------------------------------------------------------------------
_reg("count_multiples", "concise",
     "Count multiples of the given divisor in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("count_multiples", "detailed",
     "You are given a list and a divisor:\n\n{data}\n\n"
     "Count how many numbers in the list are exact multiples of the given divisor "
     "(i.e., divisible with remainder 0). "
     "Write the count as \\boxed{{answer}}.")

_reg("count_multiples", "few_shot",
     "Example 1: list=[2, 3, 4, 6], divisor=2 → multiples of 2: 2,4,6 → count=3 → \\boxed{{3}}\n"
     "Example 2: list=[1, 5, 7], divisor=3 → no multiples → count=0 → \\boxed{{0}}\n\n"
     "Now count: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# local_maxima_count
# ---------------------------------------------------------------------------
_reg("local_maxima_count", "concise",
     "Count the local maxima in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("local_maxima_count", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "A local maximum is an element that is strictly greater than both its neighbors. "
     "Count the number of local maxima (elements at the boundary are not counted unless "
     "the problem definition includes them). "
     "Write the count as \\boxed{{answer}}.")

_reg("local_maxima_count", "few_shot",
     "Example 1: [1, 3, 2, 5, 4] → 3>1,2 (yes); 5>2,4 (yes) → count=2 → \\boxed{{2}}\n"
     "Example 2: [1, 2, 3] → 2 has right neighbor 3>2 (no); no local maxima → count=0 → \\boxed{{0}}\n\n"
     "Now count local maxima in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# weighted_sum
# ---------------------------------------------------------------------------
_reg("weighted_sum", "concise",
     "Compute the weighted sum of values and weights: {data}\nAnswer as \\boxed{{answer}}.")

_reg("weighted_sum", "detailed",
     "You are given values and their corresponding weights:\n\n{data}\n\n"
     "Compute the weighted sum: multiply each value by its weight, then add all products together. "
     "Write the result as \\boxed{{answer}}.")

_reg("weighted_sum", "few_shot",
     "Example 1: values=[1,2,3], weights=[1,1,1] → 1+2+3=6 → \\boxed{{6}}\n"
     "Example 2: values=[2,4], weights=[3,1] → 2×3 + 4×1 = 10 → \\boxed{{10}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# running_average
# ---------------------------------------------------------------------------
_reg("running_average", "concise",
     "Compute the running averages for: {data}\nAnswer as \\boxed{{answer}}.")

_reg("running_average", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Compute the running average: for each position i, the running average is the mean of "
     "all elements from index 0 to i (inclusive). Provide the final running average list "
     "(or the final value if a single number is requested). "
     "Write the answer as \\boxed{{answer}}.")

_reg("running_average", "few_shot",
     "Example 1: [2, 4, 6] → running avgs: [2.0, 3.0, 4.0] → \\boxed{{[2.0, 3.0, 4.0]}}\n"
     "Example 2: [1, 3] → [1.0, 2.0] → \\boxed{{[1.0, 2.0]}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# parity_check
# ---------------------------------------------------------------------------
_reg("parity_check", "concise",
     "Check the parity of each number in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("parity_check", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "For each number, determine its parity: 'even' or 'odd'. "
     "Return the list of parities in order. "
     "Write the result as \\boxed{{answer}}.")

_reg("parity_check", "few_shot",
     "Example 1: [2, 3, 4] → ['even', 'odd', 'even'] → \\boxed{{['even', 'odd', 'even']}}\n"
     "Example 2: [7] → ['odd'] → \\boxed{{['odd']}}\n\n"
     "Now check: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# cumulative_sum
# ---------------------------------------------------------------------------
_reg("cumulative_sum", "concise",
     "Compute the cumulative sum of: {data}\nAnswer as \\boxed{{answer}}.")

_reg("cumulative_sum", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Compute the cumulative sum (prefix sums): each element in the result is the sum of "
     "all elements up to and including that position. "
     "Write the result list as \\boxed{{answer}}.")

_reg("cumulative_sum", "few_shot",
     "Example 1: [1, 2, 3] → [1, 3, 6] → \\boxed{{[1, 3, 6]}}\n"
     "Example 2: [5, -1, 2] → [5, 4, 6] → \\boxed{{[5, 4, 6]}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# reverse_list
# ---------------------------------------------------------------------------
_reg("reverse_list", "concise",
     "Reverse this list: {data}\nAnswer as \\boxed{{answer}}.")

_reg("reverse_list", "detailed",
     "Given this list:\n\n{data}\n\n"
     "Reverse the order of elements (last becomes first, first becomes last). "
     "Write the reversed list as \\boxed{{answer}}.")

_reg("reverse_list", "few_shot",
     "Example 1: [1, 2, 3] → reversed: [3, 2, 1] → \\boxed{{[3, 2, 1]}}\n"
     "Example 2: [5, -1] → reversed: [-1, 5] → \\boxed{{[-1, 5]}}\n\n"
     "Now reverse: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# rotate_list
# ---------------------------------------------------------------------------
_reg("rotate_list", "concise",
     "Rotate the list by the given positions: {data}\nAnswer as \\boxed{{answer}}.")

_reg("rotate_list", "detailed",
     "Given a list and rotation amount:\n\n{data}\n\n"
     "Rotate the list to the right by the specified number of positions "
     "(elements shifted off the right end reappear at the left). "
     "Write the rotated list as \\boxed{{answer}}.")

_reg("rotate_list", "few_shot",
     "Example 1: [1, 2, 3, 4], rotate by 1 → [4, 1, 2, 3] → \\boxed{{[4, 1, 2, 3]}}\n"
     "Example 2: [1, 2, 3], rotate by 2 → [2, 3, 1] → \\boxed{{[2, 3, 1]}}\n\n"
     "Now rotate: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# interleave_lists
# ---------------------------------------------------------------------------
_reg("interleave_lists", "concise",
     "Interleave the two lists (alternating elements): {data}\nAnswer as \\boxed{{answer}}.")

_reg("interleave_lists", "detailed",
     "Given two lists:\n\n{data}\n\n"
     "Interleave them by alternating elements: take one from the first list, one from the second, "
     "and repeat. If lists have different lengths, append remaining elements at the end. "
     "Write the interleaved list as \\boxed{{answer}}.")

_reg("interleave_lists", "few_shot",
     "Example 1: [1, 2], [3, 4] → [1, 3, 2, 4] → \\boxed{{[1, 3, 2, 4]}}\n"
     "Example 2: [1], [2, 3] → [1, 2, 3] → \\boxed{{[1, 2, 3]}}\n\n"
     "Now interleave: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# set_intersection
# ---------------------------------------------------------------------------
_reg("set_intersection", "concise",
     "Find the intersection of the two sets/lists: {data}\nAnswer as \\boxed{{answer}}.")

_reg("set_intersection", "detailed",
     "Given two lists:\n\n{data}\n\n"
     "Find all elements that appear in both lists (set intersection). "
     "Return the result as a sorted list of unique common elements. "
     "Write it as \\boxed{{answer}}.")

_reg("set_intersection", "few_shot",
     "Example 1: [1,2,3], [2,3,4] → intersection: [2,3] → \\boxed{{[2, 3]}}\n"
     "Example 2: [1,5], [2,3] → intersection: [] → \\boxed{{[]}}\n\n"
     "Now find intersection of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# set_difference
# ---------------------------------------------------------------------------
_reg("set_difference", "concise",
     "Find the set difference (elements in first but not second): {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("set_difference", "detailed",
     "Given two lists:\n\n{data}\n\n"
     "Compute the set difference: find all elements that are in the first list but NOT in the second list. "
     "Return as a sorted list of unique elements. "
     "Write the result as \\boxed{{answer}}.")

_reg("set_difference", "few_shot",
     "Example 1: [1,2,3], [2,3,4] → difference: [1] → \\boxed{{[1]}}\n"
     "Example 2: [5,6], [5,6,7] → difference: [] → \\boxed{{[]}}\n\n"
     "Now find difference: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------
_reg("moving_average", "concise",
     "Compute the moving average with the given window size: {data}\nAnswer as \\boxed{{answer}}.")

_reg("moving_average", "detailed",
     "Given a list and window size:\n\n{data}\n\n"
     "Compute the moving average: for each window of the given size, compute the mean of "
     "elements in that window. The first result corresponds to the first full window. "
     "Write the result list as \\boxed{{answer}}.")

_reg("moving_average", "few_shot",
     "Example 1: [1,2,3,4], window=2 → [(1+2)/2, (2+3)/2, (3+4)/2] = [1.5, 2.5, 3.5] → \\boxed{{[1.5, 2.5, 3.5]}}\n"
     "Example 2: [2,4,6], window=3 → [(2+4+6)/3] = [4.0] → \\boxed{{[4.0]}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# element_frequency
# ---------------------------------------------------------------------------
_reg("element_frequency", "concise",
     "Find the frequency of each element in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("element_frequency", "detailed",
     "Given this list:\n\n{data}\n\n"
     "Count how many times each distinct element appears. Return the result as a dictionary "
     "mapping each element to its count, ordered by element value. "
     "Write it as \\boxed{{answer}}.")

_reg("element_frequency", "few_shot",
     "Example 1: [1, 2, 1, 3] → count each: 1 appears 2x, 2 appears 1x, 3 appears 1x\n"
     "Example 2: [5, 5, 5] → 5 appears 3x\n\n"
     "Now find frequencies in: {data}\n"
     "Return a dict mapping element to count.\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# second_minimum
# ---------------------------------------------------------------------------
_reg("second_minimum", "concise",
     "Find the second smallest value in: {data}\nAnswer as \\boxed{{answer}}.")

_reg("second_minimum", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Find the second minimum: the smallest value that is strictly greater than the minimum. "
     "Write it as \\boxed{{answer}}.")

_reg("second_minimum", "few_shot",
     "Example 1: [1, 3, 2, 5] → min=1, 2nd min=2 → \\boxed{{2}}\n"
     "Example 2: [7, 3, 9] → 2nd min=7 → \\boxed{{7}}\n\n"
     "Now find the second minimum of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# variance
# ---------------------------------------------------------------------------
_reg("variance", "concise",
     "Compute the variance of: {data}\nRound to 2 decimal places. Answer as \\boxed{{answer}}.")

_reg("variance", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Compute the population variance: the mean of squared deviations from the mean. "
     "Formula: variance = (1/n) × Σ(xᵢ - mean)². "
     "Round to 2 decimal places. "
     "Write the result as \\boxed{{answer}}.")

_reg("variance", "few_shot",
     "Example 1: [2, 4, 6] → mean=4, deviations: [-2,0,2], sq: [4,0,4], variance=8/3≈2.67 → \\boxed{{2.67}}\n"
     "Example 2: [1, 1, 1] → mean=1, variance=0.0 → \\boxed{{0.0}}\n\n"
     "Now compute variance of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# standard_deviation
# ---------------------------------------------------------------------------
_reg("standard_deviation", "concise",
     "Compute the standard deviation of: {data}\nRound to 2 decimal places. Answer as \\boxed{{answer}}.")

_reg("standard_deviation", "detailed",
     "Given this list of numbers:\n\n{data}\n\n"
     "Compute the population standard deviation: the square root of the variance. "
     "Formula: std = sqrt((1/n) × Σ(xᵢ - mean)²). "
     "Round to 2 decimal places. "
     "Write the result as \\boxed{{answer}}.")

_reg("standard_deviation", "few_shot",
     "Example 1: [2, 4, 6] → variance≈2.67, std≈1.63 → \\boxed{{1.63}}\n"
     "Example 2: [5, 5, 5] → std=0.0 → \\boxed{{0.0}}\n\n"
     "Now compute std of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------
_reg("dot_product", "concise",
     "Compute the dot product of the two vectors: {data}\nAnswer as \\boxed{{answer}}.")

_reg("dot_product", "detailed",
     "Given two vectors:\n\n{data}\n\n"
     "Compute the dot product: multiply corresponding elements and sum the results. "
     "Formula: a·b = Σ aᵢ × bᵢ. "
     "Write the result as \\boxed{{answer}}.")

_reg("dot_product", "few_shot",
     "Example 1: [1,2,3] · [4,5,6] = 1×4 + 2×5 + 3×6 = 4+10+18 = 32 → \\boxed{{32}}\n"
     "Example 2: [1,0] · [3,4] = 3+0 = 3 → \\boxed{{3}}\n\n"
     "Now compute dot product of: {data}\n"
     "Answer: \\boxed{{answer}}")

# ===========================================================================
# MEDIUM TASKS
# ===========================================================================

# ---------------------------------------------------------------------------
# fibonacci_sequence
# ---------------------------------------------------------------------------
_reg("fibonacci_sequence", "concise",
     "Complete the Fibonacci sequence: {data}\nAnswer as \\boxed{{answer}}.")

_reg("fibonacci_sequence", "detailed",
     "You are given a partial sequence:\n\n{data}\n\n"
     "This follows the Fibonacci rule: each term equals the sum of the two preceding terms. "
     "Identify the next term. "
     "Write the answer as \\boxed{{answer}}.")

_reg("fibonacci_sequence", "few_shot",
     "Fibonacci rule: each term = sum of the two preceding terms.\n"
     "Example 1: 1, 1, 2, 3, ? → next = 3+2 = 5 → \\boxed{{5}}\n"
     "Example 2: 3, 5, 8, ? → next = 5+8 = 13 → \\boxed{{13}}\n\n"
     "Now complete: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# algebraic_sequence
# ---------------------------------------------------------------------------
_reg("algebraic_sequence", "concise",
     "Identify the algebraic pattern and find the next term: {data}\nAnswer as \\boxed{{answer}}.")

_reg("algebraic_sequence", "detailed",
     "You are given the following algebraic sequence:\n\n{data}\n\n"
     "Analyze the mathematical relationship between consecutive terms. "
     "It may involve multiplication, powers, or polynomial rules. "
     "Find the next term and write it as \\boxed{{answer}}.")

_reg("algebraic_sequence", "few_shot",
     "Example 1: 2, 4, 8, 16, ? → each term doubled → 32 → \\boxed{{32}}\n"
     "Example 2: 1, 4, 9, 16, ? → perfect squares → 25 → \\boxed{{25}}\n\n"
     "Now complete: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# geometric_sequence
# ---------------------------------------------------------------------------
_reg("geometric_sequence", "concise",
     "Find the next term in the geometric sequence: {data}\nAnswer as \\boxed{{answer}}.")

_reg("geometric_sequence", "detailed",
     "You are given a geometric sequence (each term is multiplied by a constant ratio):\n\n{data}\n\n"
     "Find the common ratio r, then compute the next term = last term × r. "
     "Write the answer as \\boxed{{answer}}.")

_reg("geometric_sequence", "few_shot",
     "Example 1: 2, 6, 18, ? → ratio=3, next=54 → \\boxed{{54}}\n"
     "Example 2: 100, 10, 1, ? → ratio=0.1, next=0.1 → \\boxed{{0.1}}\n\n"
     "Now complete: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# prime_sequence
# ---------------------------------------------------------------------------
_reg("prime_sequence", "concise",
     "Find the next term in the prime-based sequence: {data}\nAnswer as \\boxed{{answer}}.")

_reg("prime_sequence", "detailed",
     "You are given a sequence based on prime numbers:\n\n{data}\n\n"
     "Analyze the pattern (consecutive primes, prime-derived formula, etc.) and find the next term. "
     "Write the answer as \\boxed{{answer}}.")

_reg("prime_sequence", "few_shot",
     "Example 1: 2, 3, 5, 7, ? → sequence of primes → 11 → \\boxed{{11}}\n"
     "Example 2: 4, 9, 25, 49, ? → squares of primes → 121 → \\boxed{{121}}\n\n"
     "Now complete: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# complex_pattern
# ---------------------------------------------------------------------------
_reg("complex_pattern", "concise",
     "Find the next term in the complex pattern: {data}\nAnswer as \\boxed{{answer}}.")

_reg("complex_pattern", "detailed",
     "You are given a complex sequence:\n\n{data}\n\n"
     "The pattern may combine arithmetic, geometric, or other rules (e.g., alternating rules, "
     "interleaved sequences, or multi-step formulas). "
     "Carefully analyze all relationships between terms and find the next one. "
     "Write your answer as \\boxed{{answer}}.")

_reg("complex_pattern", "few_shot",
     "Example 1: 1, 2, 3, 6, 11, ? → each term = sum of 3 preceding → 20 → \\boxed{{20}}\n"
     "Example 2: 1, 1, 2, 4, 7, ? → differences: 0,1,2,3 → next diff=4, answer=11 → \\boxed{{11}}\n\n"
     "Now complete: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# arithmetic_progression
# ---------------------------------------------------------------------------
_reg("arithmetic_progression", "concise",
     "Find the next term in the arithmetic progression: {data}\nAnswer as \\boxed{{answer}}.")

_reg("arithmetic_progression", "detailed",
     "You are given an arithmetic progression (constant difference between consecutive terms):\n\n{data}\n\n"
     "Find the common difference d, then compute next term = last term + d. "
     "Write the answer as \\boxed{{answer}}.")

_reg("arithmetic_progression", "few_shot",
     "Example 1: 3, 7, 11, ? → d=4, next=15 → \\boxed{{15}}\n"
     "Example 2: 10, 7, 4, ? → d=-3, next=1 → \\boxed{{1}}\n\n"
     "Now complete: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# harmonic_sequence
# ---------------------------------------------------------------------------
_reg("harmonic_sequence", "concise",
     "Find the next term in the harmonic sequence: {data}\nAnswer as \\boxed{{answer}}.")

_reg("harmonic_sequence", "detailed",
     "You are given a harmonic sequence (reciprocals of an arithmetic progression):\n\n{data}\n\n"
     "The denominators form an arithmetic progression. Find the next denominator and take its reciprocal. "
     "Write the answer as \\boxed{{answer}}.")

_reg("harmonic_sequence", "few_shot",
     "Example 1: 1/1, 1/2, 1/3, ? → denominators: 1,2,3,4 → answer=1/4 → \\boxed{{1/4}}\n"
     "Example 2: 1/2, 1/4, 1/6, ? → denominators: 2,4,6,8 → answer=1/8 → \\boxed{{1/8}}\n\n"
     "Now complete: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# collatz_sequence
# ---------------------------------------------------------------------------
_reg("collatz_sequence", "concise",
     "Find the next Collatz term: {data}\n"
     "(If even: n/2; if odd: 3n+1). Answer as \\boxed{{answer}}.")

_reg("collatz_sequence", "detailed",
     "You are given a partial Collatz sequence:\n\n{data}\n\n"
     "Collatz rule: if the current term is even, the next term is half of it; "
     "if the current term is odd, the next term is 3 times it plus 1. "
     "Find the next term. Write it as \\boxed{{answer}}.")

_reg("collatz_sequence", "few_shot",
     "Example 1: 6, 3, ? → 3 is odd, 3×3+1=10 → \\boxed{{10}}\n"
     "Example 2: 8, 4, ? → 4 is even, 4/2=2 → \\boxed{{2}}\n\n"
     "Now find the next term in: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# polynomial_evaluation
# ---------------------------------------------------------------------------
_reg("polynomial_evaluation", "concise",
     "Evaluate the polynomial at the given x value: {data}\nAnswer as \\boxed{{answer}}.")

_reg("polynomial_evaluation", "detailed",
     "You are given a polynomial and a value of x:\n\n{data}\n\n"
     "Substitute the value of x into the polynomial and compute the result. "
     "Show intermediate steps if helpful. "
     "Write the final value as \\boxed{{answer}}.")

_reg("polynomial_evaluation", "few_shot",
     "Example 1: p(x) = x² + 2x + 1, x=3 → 9 + 6 + 1 = 16 → \\boxed{{16}}\n"
     "Example 2: p(x) = 2x - 3, x=5 → 10 - 3 = 7 → \\boxed{{7}}\n\n"
     "Now evaluate: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# matrix_operations
# ---------------------------------------------------------------------------
_reg("matrix_operations", "concise",
     "Perform the matrix operation: {data}\nAnswer as \\boxed{{answer}}.")

_reg("matrix_operations", "detailed",
     "You are given matrices and a matrix operation:\n\n{data}\n\n"
     "Perform the specified operation (addition, subtraction, or multiplication). "
     "Show your work step by step. "
     "Write the resulting matrix as \\boxed{{answer}}.")

_reg("matrix_operations", "few_shot",
     "Example 1: A=[[1,2],[3,4]], B=[[5,6],[7,8]], add → [[6,8],[10,12]] → \\boxed{{[[6,8],[10,12]]}}\n"
     "Example 2: A=[[1,2],[3,4]], B=[[1,0],[0,1]], multiply → [[1,2],[3,4]] → \\boxed{{[[1,2],[3,4]]}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# number_base_conversion
# ---------------------------------------------------------------------------
_reg("number_base_conversion", "concise",
     "Convert the number as specified: {data}\nAnswer as \\boxed{{answer}}.")

_reg("number_base_conversion", "detailed",
     "You are given a number and a base conversion task:\n\n{data}\n\n"
     "Perform the conversion (e.g., decimal to binary, decimal to hex, binary to decimal, etc.). "
     "Show your conversion steps. "
     "Write the result as \\boxed{{answer}}.")

_reg("number_base_conversion", "few_shot",
     "Example 1: Convert 10 (decimal) to binary → 1010 → \\boxed{{1010}}\n"
     "Example 2: Convert 255 (decimal) to hex → FF → \\boxed{{FF}}\n\n"
     "Now convert: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# logical_operations
# ---------------------------------------------------------------------------
_reg("logical_operations", "concise",
     "Evaluate the logical expression: {data}\nAnswer as \\boxed{{answer}}.")

_reg("logical_operations", "detailed",
     "You are given variable assignments and a logical expression:\n\n{data}\n\n"
     "Substitute the variable values and evaluate the expression using boolean logic rules "
     "(AND, OR, NOT, XOR). "
     "Write True or False as \\boxed{{answer}}.")

_reg("logical_operations", "few_shot",
     "Example 1: A=True, B=False, A AND B → False → \\boxed{{False}}\n"
     "Example 2: A=True, B=False, A OR B → True → \\boxed{{True}}\n\n"
     "Now evaluate: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# pattern_completion
# ---------------------------------------------------------------------------
_reg("pattern_completion", "concise",
     "Find the next number in this pattern: {data}\nAnswer as \\boxed{{answer}}.")

_reg("pattern_completion", "detailed",
     "You are given a numeric pattern:\n\n{data}\n\n"
     "Identify the underlying rule (perfect squares, cubes, triangular numbers, powers, etc.) "
     "and find the next term. "
     "Write the answer as \\boxed{{answer}}.")

_reg("pattern_completion", "few_shot",
     "Example 1: 1, 4, 9, 16, ? → perfect squares → 25 → \\boxed{{25}}\n"
     "Example 2: 1, 8, 27, 64, ? → perfect cubes → 125 → \\boxed{{125}}\n\n"
     "Now complete: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# gcd_lcm
# ---------------------------------------------------------------------------
_reg("gcd_lcm", "concise",
     "Compute the GCD/LCM as specified: {data}\nAnswer as \\boxed{{answer}}.")

_reg("gcd_lcm", "detailed",
     "You are given numbers and a task (GCD or LCM):\n\n{data}\n\n"
     "For GCD (Greatest Common Divisor): find the largest integer that divides all numbers exactly.\n"
     "For LCM (Least Common Multiple): find the smallest positive integer divisible by all numbers.\n"
     "Write the result as \\boxed{{answer}}.")

_reg("gcd_lcm", "few_shot",
     "Example 1: GCD(12, 18) = 6 → \\boxed{{6}}\n"
     "Example 2: LCM(4, 6) = 12 → \\boxed{{12}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# combinatorics
# ---------------------------------------------------------------------------
_reg("combinatorics", "concise",
     "Compute the combinatorics value: {data}\nAnswer as \\boxed{{answer}}.")

_reg("combinatorics", "detailed",
     "You are given a combinatorics problem:\n\n{data}\n\n"
     "For permutations P(n,r) = n! / (n-r)!\n"
     "For combinations C(n,r) = n! / (r! × (n-r)!)\n"
     "Compute the value and write it as \\boxed{{answer}}.")

_reg("combinatorics", "few_shot",
     "Example 1: C(5,2) = 5!/(2!×3!) = 10 → \\boxed{{10}}\n"
     "Example 2: P(4,2) = 4!/2! = 12 → \\boxed{{12}}\n\n"
     "Now compute: {data}\n"
     "Answer: \\boxed{{answer}}")

# ===========================================================================
# HARD TASKS
# ===========================================================================

# ---------------------------------------------------------------------------
# tower_hanoi
# ---------------------------------------------------------------------------
_reg("tower_hanoi", "concise",
     "Solve the Tower of Hanoi puzzle: {data}\n"
     "List the minimum number of moves required. Answer as \\boxed{{answer}}.")

_reg("tower_hanoi", "detailed",
     "You are given a Tower of Hanoi problem:\n\n{data}\n\n"
     "The minimum number of moves to solve a Tower of Hanoi with n disks is 2ⁿ - 1. "
     "Count the disks and compute the minimum moves. "
     "Then list the sequence of moves (from peg → to peg). "
     "Write the minimum move count as \\boxed{{answer}}.")

_reg("tower_hanoi", "few_shot",
     "Example 1: 2 disks → min moves = 3 → \\boxed{{3}}\n"
     "Example 2: 3 disks → min moves = 7 → \\boxed{{7}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# n_queens
# ---------------------------------------------------------------------------
_reg("n_queens", "concise",
     "Solve the N-Queens problem: {data}\n"
     "Provide a valid placement as \\boxed{{answer}}.")

_reg("n_queens", "detailed",
     "You are given an N-Queens problem:\n\n{data}\n\n"
     "Place N queens on an N×N chessboard so that no two queens share the same row, column, or diagonal. "
     "Represent the solution as a list of column positions (0-indexed) for each row. "
     "Write your answer as \\boxed{{answer}}.")

_reg("n_queens", "few_shot",
     "Example: 4-Queens → one valid solution: [1, 3, 0, 2] "
     "(queen in row 0 at col 1, row 1 at col 3, etc.) → \\boxed{{[1, 3, 0, 2]}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# graph_coloring
# ---------------------------------------------------------------------------
_reg("graph_coloring", "concise",
     "Find the chromatic number and a valid coloring for the graph: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("graph_coloring", "detailed",
     "You are given a graph coloring problem:\n\n{data}\n\n"
     "Assign colors to vertices such that no two adjacent vertices share the same color, "
     "using the minimum number of colors (the chromatic number). "
     "Provide the coloring assignment. "
     "Write the chromatic number as \\boxed{{answer}}.")

_reg("graph_coloring", "few_shot",
     "Example: triangle graph (3 vertices, all connected) → chromatic number = 3 → \\boxed{{3}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# boolean_sat
# ---------------------------------------------------------------------------
_reg("boolean_sat", "concise",
     "Solve this SAT problem (find satisfying assignment): {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("boolean_sat", "detailed",
     "You are given a Boolean Satisfiability problem:\n\n{data}\n\n"
     "Find an assignment of True/False values to variables that satisfies all clauses. "
     "Each clause is a disjunction of literals. All clauses must be satisfied simultaneously. "
     "Write the satisfying assignment or UNSAT as \\boxed{{answer}}.")

_reg("boolean_sat", "few_shot",
     "Example: clauses: (x1 OR x2), (NOT x1 OR x3) → x1=True, x2=True, x3=True satisfies all "
     "→ \\boxed{{x1=True, x2=True, x3=True}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# sudoku_solving
# ---------------------------------------------------------------------------
_reg("sudoku_solving", "concise",
     "Solve the Sudoku puzzle: {data}\nAnswer as \\boxed{{answer}}.")

_reg("sudoku_solving", "detailed",
     "You are given a Sudoku puzzle:\n\n{data}\n\n"
     "Fill the empty cells (marked with 0 or .) so that every row, column, and box "
     "contains each digit exactly once. "
     "Write the completed grid as \\boxed{{answer}}.")

_reg("sudoku_solving", "few_shot",
     "Example: A 4×4 Sudoku grid where every row/col/2×2 box has digits 1-4 exactly once. "
     "[[1,0,3,0],[0,3,0,1],[3,0,1,0],[0,1,0,3]] → solution: [[1,2,3,4],[4,3,2,1],[3,4,1,2],[2,1,4,3]] "
     "→ \\boxed{{[[1,2,3,4],[4,3,2,1],[3,4,1,2],[2,1,4,3]]}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# cryptarithmetic
# ---------------------------------------------------------------------------
_reg("cryptarithmetic", "concise",
     "Solve the cryptarithmetic puzzle: {data}\nAnswer as \\boxed{{answer}}.")

_reg("cryptarithmetic", "detailed",
     "You are given a cryptarithmetic puzzle:\n\n{data}\n\n"
     "Assign a unique digit (0-9) to each letter such that the arithmetic equation holds. "
     "Leading digits cannot be zero. Find the digit assignment and verify the equation. "
     "Write the mapping as \\boxed{{answer}}.")

_reg("cryptarithmetic", "few_shot",
     "Example: SEND + MORE = MONEY → S=9,E=5,N=6,D=7,M=1,O=0,R=8,Y=2 "
     "→ \\boxed{{S=9,E=5,N=6,D=7,M=1,O=0,R=8,Y=2}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# matrix_chain_multiplication
# ---------------------------------------------------------------------------
_reg("matrix_chain_multiplication", "concise",
     "Find the minimum number of multiplications for the matrix chain: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("matrix_chain_multiplication", "detailed",
     "You are given a matrix chain multiplication problem:\n\n{data}\n\n"
     "Find the optimal parenthesization order to minimize the total number of scalar multiplications. "
     "Use dynamic programming. "
     "Write the minimum multiplication count as \\boxed{{answer}}.")

_reg("matrix_chain_multiplication", "few_shot",
     "Example: matrices with dimensions [10,30,5,60] → optimal cost = 27000 → \\boxed{{27000}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# modular_systems (modular_systems_solver)
# ---------------------------------------------------------------------------
_reg("modular_systems", "concise",
     "Solve the system of modular equations: {data}\nAnswer as \\boxed{{answer}}.")

_reg("modular_systems", "detailed",
     "You are given a system of modular arithmetic equations:\n\n{data}\n\n"
     "Find the smallest non-negative integer x satisfying all congruences. "
     "Use the Chinese Remainder Theorem if applicable. "
     "Write the solution as \\boxed{{answer}}.")

_reg("modular_systems", "few_shot",
     "Example: x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 8 → \\boxed{{8}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# constraint_optimization
# ---------------------------------------------------------------------------
_reg("constraint_optimization", "concise",
     "Solve the constraint optimization problem: {data}\nAnswer as \\boxed{{answer}}.")

_reg("constraint_optimization", "detailed",
     "You are given a constraint optimization problem:\n\n{data}\n\n"
     "Find the assignment that maximizes (or minimizes) the objective function "
     "while satisfying all given constraints. "
     "Write the optimal objective value as \\boxed{{answer}}.")

_reg("constraint_optimization", "few_shot",
     "Example: maximize revenue subject to budget and resource constraints. "
     "Select the optimal combination of projects to maximize value.\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# logic_grid_puzzles
# ---------------------------------------------------------------------------
_reg("logic_grid_puzzles", "concise",
     "Solve the logic grid puzzle: {data}\nAnswer as \\boxed{{answer}}.")

_reg("logic_grid_puzzles", "detailed",
     "You are given a logic grid puzzle:\n\n{data}\n\n"
     "Use the given clues to uniquely assign attributes to entities. "
     "Systematically eliminate impossible assignments using each clue. "
     "Write the complete assignment as \\boxed{{answer}}.")

_reg("logic_grid_puzzles", "few_shot",
     "Example: 3 people (Alice, Bob, Carol), 3 colors. Clues narrow each person to one color. "
     "→ Answer: Alice=Red, Bob=Blue, Carol=Green → \\boxed{{Alice=Red, Bob=Blue, Carol=Green}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------------
_reg("shortest_path", "concise",
     "Find the shortest path distance from source to target: {data}\nAnswer as \\boxed{{answer}}.")

_reg("shortest_path", "detailed",
     "You are given a weighted directed graph and a source/target pair:\n\n{data}\n\n"
     "Find the shortest path (minimum total weight) from source to target using Dijkstra's algorithm "
     "or another shortest-path method. "
     "Write the minimum distance as \\boxed{{answer}}.")

_reg("shortest_path", "few_shot",
     "Example: graph A→B(2), B→C(3), A→C(10). Shortest A→C = A→B→C = 5 → \\boxed{{5}}\n\n"
     "Now find shortest path: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# knapsack
# ---------------------------------------------------------------------------
_reg("knapsack", "concise",
     "Solve the 0/1 Knapsack problem: {data}\nAnswer as \\boxed{{answer}}.")

_reg("knapsack", "detailed",
     "You are given a Knapsack problem:\n\n{data}\n\n"
     "Select a subset of items to maximize total value without exceeding the weight capacity. "
     "Each item can be taken at most once (0/1 knapsack). "
     "Use dynamic programming. "
     "Write the maximum achievable value as \\boxed{{answer}}.")

_reg("knapsack", "few_shot",
     "Example: items=[(w=2,v=3),(w=3,v=4),(w=4,v=5)], capacity=5 → best: items 1+2, value=7 → \\boxed{{7}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# traveling_salesman
# ---------------------------------------------------------------------------
_reg("traveling_salesman", "concise",
     "Find the minimum cost Hamiltonian cycle (TSP): {data}\nAnswer as \\boxed{{answer}}.")

_reg("traveling_salesman", "detailed",
     "You are given a Traveling Salesman Problem (TSP):\n\n{data}\n\n"
     "Find the shortest route that visits every city exactly once and returns to the starting city. "
     "Try all permutations or use dynamic programming for small instances. "
     "Write the minimum total distance as \\boxed{{answer}}.")

_reg("traveling_salesman", "few_shot",
     "Example: 3 cities, distances: 0→1=10, 1→2=15, 2→0=20 → tour: 0→1→2→0 = 45 → \\boxed{{45}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# longest_common_subsequence
# ---------------------------------------------------------------------------
_reg("longest_common_subsequence", "concise",
     "Find the length of the longest common subsequence: {data}\nAnswer as \\boxed{{answer}}.")

_reg("longest_common_subsequence", "detailed",
     "You are given two strings:\n\n{data}\n\n"
     "Find the length of the Longest Common Subsequence (LCS): the longest sequence of characters "
     "that appears in both strings (not necessarily contiguous). "
     "Use dynamic programming. "
     "Write the LCS length as \\boxed{{answer}}.")

_reg("longest_common_subsequence", "few_shot",
     "Example 1: 'ABCBDAB' and 'BDCABA' → LCS = 'BCBA' or 'BDAB', length=4 → \\boxed{{4}}\n"
     "Example 2: 'ABC' and 'AC' → LCS = 'AC', length=2 → \\boxed{{2}}\n\n"
     "Now find LCS length: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# minimax_game
# ---------------------------------------------------------------------------
_reg("minimax_game", "concise",
     "Find the minimax value of the game tree: {data}\nAnswer as \\boxed{{answer}}.")

_reg("minimax_game", "detailed",
     "You are given a game tree for a two-player zero-sum game:\n\n{data}\n\n"
     "Apply the minimax algorithm: MAX nodes take the maximum of children values, "
     "MIN nodes take the minimum. "
     "Compute the root node's minimax value. "
     "Write the result as \\boxed{{answer}}.")

_reg("minimax_game", "few_shot",
     "Example: root (MAX) has children 3 and 5 → MAX chooses 5 → \\boxed{{5}}\n"
     "(If children were MIN nodes with leaves [3,6] and [2,5] → MIN(3,6)=3, MIN(2,5)=2 → MAX(3,2)=3 → \\boxed{{3}})\n\n"
     "Now compute minimax: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# regex_matching
# ---------------------------------------------------------------------------
_reg("regex_matching", "concise",
     "Does the string fully match the regex pattern? {data}\n"
     "Answer True or False as \\boxed{{answer}}.")

_reg("regex_matching", "detailed",
     "You are given a regular expression pattern and a test string:\n\n{data}\n\n"
     "Determine if the ENTIRE string fully matches the pattern (from start to end). "
     "Consider: '.' matches any character, '*' means zero or more of preceding, "
     "'^' anchors to start, '$' anchors to end, etc. "
     "Write True or False as \\boxed{{answer}}.")

_reg("regex_matching", "few_shot",
     "Example 1: pattern='a*b', string='aaab' → fully matches → \\boxed{{True}}\n"
     "Example 2: pattern='a.c', string='ac' → does not fully match → \\boxed{{False}}\n\n"
     "Now check: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# topological_sort
# ---------------------------------------------------------------------------
_reg("topological_sort", "concise",
     "Give a valid topological ordering of the DAG: {data}\nAnswer as \\boxed{{answer}}.")

_reg("topological_sort", "detailed",
     "You are given a Directed Acyclic Graph (DAG):\n\n{data}\n\n"
     "Find a topological ordering: an ordering of vertices such that for every directed edge u→v, "
     "vertex u appears before vertex v. "
     "Write one valid topological order as \\boxed{{answer}}.")

_reg("topological_sort", "few_shot",
     "Example: nodes=[1,2,3], edges=[(1,2),(1,3),(2,3)] → one valid order: [1,2,3] → \\boxed{{[1, 2, 3]}}\n\n"
     "Now find topological order: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# interval_scheduling
# ---------------------------------------------------------------------------
_reg("interval_scheduling", "concise",
     "Find the maximum number of non-overlapping intervals: {data}\nAnswer as \\boxed{{answer}}.")

_reg("interval_scheduling", "detailed",
     "You are given a set of intervals with start and end times:\n\n{data}\n\n"
     "Find the maximum number of non-overlapping intervals you can select "
     "(the interval scheduling maximization problem). "
     "Use a greedy approach: sort by end time, greedily select non-overlapping intervals. "
     "Write the maximum count as \\boxed{{answer}}.")

_reg("interval_scheduling", "few_shot",
     "Example: intervals [(1,3),(2,4),(3,5)] → select (1,3),(3,5): 2 non-overlapping → \\boxed{{2}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# coin_change
# ---------------------------------------------------------------------------
_reg("coin_change", "concise",
     "Find the minimum number of coins to make the target amount: {data}\nAnswer as \\boxed{{answer}}.")

_reg("coin_change", "detailed",
     "You are given a Coin Change problem:\n\n{data}\n\n"
     "Find the minimum number of coins from the given denominations needed to sum to the target amount. "
     "You can use any coin denomination an unlimited number of times. "
     "If impossible, return -1. "
     "Use dynamic programming. "
     "Write the minimum coin count as \\boxed{{answer}}.")

_reg("coin_change", "few_shot",
     "Example 1: coins=[1,5,10], target=11 → 10+1=2 coins → \\boxed{{2}}\n"
     "Example 2: coins=[2], target=3 → impossible → \\boxed{{-1}}\n\n"
     "Now solve: {data}\n"
     "Answer: \\boxed{{answer}}")

# ---------------------------------------------------------------------------
# edit_distance
# ---------------------------------------------------------------------------
_reg("edit_distance", "concise",
     "Compute the edit (Levenshtein) distance between the two strings: {data}\n"
     "Answer as \\boxed{{answer}}.")

_reg("edit_distance", "detailed",
     "You are given two strings:\n\n{data}\n\n"
     "Compute the Edit Distance (Levenshtein distance): the minimum number of single-character "
     "operations (insertions, deletions, substitutions) needed to transform the first string "
     "into the second string. "
     "Use dynamic programming. "
     "Write the edit distance as \\boxed{{answer}}.")

_reg("edit_distance", "few_shot",
     "Example 1: 'kitten' → 'sitting': 3 operations → \\boxed{{3}}\n"
     "Example 2: 'cat' → 'cat': 0 operations → \\boxed{{0}}\n\n"
     "Now compute edit distance: {data}\n"
     "Answer: \\boxed{{answer}}")


# ===========================================================================
# CHAIN-OF-THOUGHT (CoT) PROMPTS — all 79 tasks
# ===========================================================================
# CoT prompts instruct the model to reason step by step before giving
# the final \\boxed{{answer}}.  This generally improves accuracy on
# harder tasks at the cost of more tokens.
# ===========================================================================

# --- EASY ---
_reg("sum", "cot",
     "Think step by step.\n\n"
     "Sum these numbers: {data}\n\n"
     "Show your reasoning, then write your final answer as \\boxed{{answer}}.")

_reg("sorting", "cot",
     "Think step by step.\n\n"
     "Sort this list in ascending order: {data}\n\n"
     "Show your reasoning, then write the sorted list as \\boxed{{answer}}.")

_reg("comparison", "cot",
     "Think step by step.\n\n"
     "Compare these two numbers: {data}\n\n"
     "Determine if the first is 'greater', 'less', or 'equal' to the second. "
     "Show your reasoning, then write your answer as \\boxed{{answer}}.")

_reg("multiplication", "cot",
     "Think step by step.\n\n"
     "Multiply these numbers: {data}\n\n"
     "Show your work, then write the product as \\boxed{{answer}}.")

_reg("odd_count", "cot",
     "Think step by step.\n\n"
     "Count the odd numbers in: {data}\n\n"
     "Check each number. Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("even_count", "cot",
     "Think step by step.\n\n"
     "Count the even numbers in: {data}\n\n"
     "Check each number. Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("absolute_difference", "cot",
     "Think step by step.\n\n"
     "Find the absolute difference between these two numbers: {data}\n\n"
     "Compute |a - b|. Show your reasoning, then write your answer as \\boxed{{answer}}.")

_reg("division", "cot",
     "Think step by step.\n\n"
     "Divide the first number by the second: {data}\n\n"
     "Round to 2 decimal places if needed. Show your work, then write the quotient as \\boxed{{answer}}.")

_reg("find_maximum", "cot",
     "Think step by step.\n\n"
     "Find the maximum value in: {data}\n\n"
     "Scan through all elements. Show your reasoning, then write the maximum as \\boxed{{answer}}.")

_reg("find_minimum", "cot",
     "Think step by step.\n\n"
     "Find the minimum value in: {data}\n\n"
     "Scan through all elements. Show your reasoning, then write the minimum as \\boxed{{answer}}.")

_reg("mean", "cot",
     "Think step by step.\n\n"
     "Calculate the mean of: {data}\n\n"
     "Sum all values and divide by count. Show your work, then write the mean as \\boxed{{answer}}.")

_reg("median", "cot",
     "Think step by step.\n\n"
     "Find the median of: {data}\n\n"
     "Sort the values first, then find the middle element. Show your reasoning, "
     "then write the median as \\boxed{{answer}}.")

_reg("mode", "cot",
     "Think step by step.\n\n"
     "Find the mode (most frequent value) in: {data}\n\n"
     "Count each value's frequency. Show your reasoning, then write the mode as \\boxed{{answer}}.")

_reg("range", "cot",
     "Think step by step.\n\n"
     "Find the range (max - min) of: {data}\n\n"
     "Identify the max and min, then subtract. Show your reasoning, "
     "then write the range as \\boxed{{answer}}.")

_reg("subtraction", "cot",
     "Think step by step.\n\n"
     "Subtract the second from the first: {data}\n\n"
     "Show your work, then write the difference as \\boxed{{answer}}.")

_reg("sum_of_digits", "cot",
     "Think step by step.\n\n"
     "Find the sum of digits of: {data}\n\n"
     "Break the number into individual digits and add them. "
     "Show your work, then write the sum as \\boxed{{answer}}.")

_reg("count_unique", "cot",
     "Think step by step.\n\n"
     "Count the unique values in: {data}\n\n"
     "Identify duplicates and count distinct elements. "
     "Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("count_negative", "cot",
     "Think step by step.\n\n"
     "Count the negative numbers in: {data}\n\n"
     "Check each number. Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("index_of_maximum", "cot",
     "Think step by step.\n\n"
     "Find the index (0-based) of the maximum value in: {data}\n\n"
     "Scan through all elements. Show your reasoning, "
     "then write the index as \\boxed{{answer}}.")

_reg("second_maximum", "cot",
     "Think step by step.\n\n"
     "Find the second largest value in: {data}\n\n"
     "Identify the maximum, then find the next largest distinct value. "
     "Show your reasoning, then write it as \\boxed{{answer}}.")

_reg("second_minimum", "cot",
     "Think step by step.\n\n"
     "Find the second smallest value in: {data}\n\n"
     "Identify the minimum, then find the next smallest distinct value. "
     "Show your reasoning, then write it as \\boxed{{answer}}.")

_reg("reverse_list", "cot",
     "Think step by step.\n\n"
     "Reverse this list: {data}\n\n"
     "Show your reasoning, then write the reversed list as \\boxed{{answer}}.")

_reg("cumulative_sum", "cot",
     "Think step by step.\n\n"
     "Compute the cumulative sum of: {data}\n\n"
     "At each position, sum all elements from the start up to that position. "
     "Show your work, then write the cumulative sum list as \\boxed{{answer}}.")

_reg("alternating_sum", "cot",
     "Think step by step.\n\n"
     "Compute the alternating sum of: {data}\n\n"
     "Alternate between adding (+) and subtracting (-) each element starting with +. "
     "Show your work, then write the result as \\boxed{{answer}}.")

_reg("dot_product", "cot",
     "Think step by step.\n\n"
     "Compute the dot product of these two vectors: {data}\n\n"
     "Multiply corresponding elements and sum. "
     "Show your work, then write the dot product as \\boxed{{answer}}.")

_reg("weighted_sum", "cot",
     "Think step by step.\n\n"
     "Compute the weighted sum: {data}\n\n"
     "Multiply each value by its weight and sum. "
     "Show your work, then write the result as \\boxed{{answer}}.")

_reg("parity_check", "cot",
     "Think step by step.\n\n"
     "Determine if this number is even or odd: {data}\n\n"
     "Check divisibility by 2. Show your reasoning, "
     "then write 'even' or 'odd' as \\boxed{{answer}}.")

_reg("count_multiples", "cot",
     "Think step by step.\n\n"
     "Count the multiples of k in this list: {data}\n\n"
     "Check each number for divisibility. Show your reasoning, "
     "then write the count as \\boxed{{answer}}.")

_reg("count_perfect_squares", "cot",
     "Think step by step.\n\n"
     "Count the perfect squares in: {data}\n\n"
     "Check each number. Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("count_palindromic", "cot",
     "Think step by step.\n\n"
     "Count the palindromic numbers in: {data}\n\n"
     "A number is palindromic if it reads the same forwards and backwards. "
     "Check each number. Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("local_maxima_count", "cot",
     "Think step by step.\n\n"
     "Count the local maxima in: {data}\n\n"
     "A local maximum is an element greater than both its neighbors. "
     "Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("max_adjacent_difference", "cot",
     "Think step by step.\n\n"
     "Find the maximum adjacent difference in: {data}\n\n"
     "Compute |a[i+1] - a[i]| for each pair and find the maximum. "
     "Show your reasoning, then write the result as \\boxed{{answer}}.")

_reg("longest_increasing_subsequence", "cot",
     "Think step by step.\n\n"
     "Find the length of the longest strictly increasing subsequence in: {data}\n\n"
     "Use dynamic programming. Show your reasoning, "
     "then write the length as \\boxed{{answer}}.")

_reg("count_greater_than_previous", "cot",
     "Think step by step.\n\n"
     "Count how many elements are greater than their preceding element in: {data}\n\n"
     "Compare each element with the one before it. "
     "Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("sum_of_max_indices", "cot",
     "Think step by step.\n\n"
     "Find all indices where the maximum value occurs, then sum those indices: {data}\n\n"
     "Show your reasoning, then write the sum of indices as \\boxed{{answer}}.")

_reg("running_average", "cot",
     "Think step by step.\n\n"
     "Compute the running average at each position: {data}\n\n"
     "At position i, average all elements from index 0 to i. "
     "Show your work, then write the list of running averages as \\boxed{{answer}}.")

_reg("moving_average", "cot",
     "Think step by step.\n\n"
     "Compute the moving average: {data}\n\n"
     "For each valid window, average the elements in that window. "
     "Show your work, then write the result as \\boxed{{answer}}.")

_reg("variance", "cot",
     "Think step by step.\n\n"
     "Calculate the population variance of: {data}\n\n"
     "First find the mean, then average the squared deviations. "
     "Show your work, then write the variance as \\boxed{{answer}}.")

_reg("standard_deviation", "cot",
     "Think step by step.\n\n"
     "Calculate the population standard deviation of: {data}\n\n"
     "First find the variance, then take the square root. "
     "Show your work, then write the standard deviation as \\boxed{{answer}}.")

_reg("element_frequency", "cot",
     "Think step by step.\n\n"
     "Find the frequency of each element in: {data}\n\n"
     "Count occurrences of each value. Show your reasoning, "
     "then write the frequency dictionary as \\boxed{{answer}}.")

_reg("set_intersection", "cot",
     "Think step by step.\n\n"
     "Find the intersection of these two sets: {data}\n\n"
     "Identify elements present in both sets. "
     "Show your reasoning, then write the intersection as \\boxed{{answer}}.")

_reg("set_difference", "cot",
     "Think step by step.\n\n"
     "Find the set difference (first minus second): {data}\n\n"
     "Identify elements in the first set but not the second. "
     "Show your reasoning, then write the difference as \\boxed{{answer}}.")

_reg("interleave_lists", "cot",
     "Think step by step.\n\n"
     "Interleave these two lists: {data}\n\n"
     "Alternate elements from each list. "
     "Show your reasoning, then write the interleaved list as \\boxed{{answer}}.")

_reg("rotate_list", "cot",
     "Think step by step.\n\n"
     "Rotate this list by the given positions: {data}\n\n"
     "Show your reasoning, then write the rotated list as \\boxed{{answer}}.")

# --- MEDIUM ---
_reg("fibonacci_sequence", "cot",
     "Think step by step.\n\n"
     "Compute the requested Fibonacci value: {data}\n\n"
     "Generate the sequence starting from F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2). "
     "Show your work, then write your answer as \\boxed{{answer}}.")

_reg("prime_sequence", "cot",
     "Think step by step.\n\n"
     "Find the requested prime numbers: {data}\n\n"
     "Check divisibility for each candidate. "
     "Show your reasoning, then write your answer as \\boxed{{answer}}.")

_reg("geometric_sequence", "cot",
     "Think step by step.\n\n"
     "Solve this geometric sequence problem: {data}\n\n"
     "Use the formula a_n = a * r^(n-1). "
     "Show your work, then write your answer as \\boxed{{answer}}.")

_reg("arithmetic_progression", "cot",
     "Think step by step.\n\n"
     "Solve this arithmetic progression problem: {data}\n\n"
     "Use the formula a_n = a + (n-1)*d. "
     "Show your work, then write your answer as \\boxed{{answer}}.")

_reg("harmonic_sequence", "cot",
     "Think step by step.\n\n"
     "Compute the harmonic series value: {data}\n\n"
     "Sum 1/1 + 1/2 + 1/3 + ... "
     "Show your work, then write your answer as \\boxed{{answer}}.")

_reg("collatz_sequence", "cot",
     "Think step by step.\n\n"
     "Compute the Collatz sequence: {data}\n\n"
     "If even, divide by 2. If odd, multiply by 3 and add 1. Repeat until 1. "
     "Show each step, then write your answer as \\boxed{{answer}}.")

_reg("gcd_lcm", "cot",
     "Think step by step.\n\n"
     "Compute GCD and/or LCM: {data}\n\n"
     "Use the Euclidean algorithm. "
     "Show your work, then write your answer as \\boxed{{answer}}.")

_reg("polynomial_evaluation", "cot",
     "Think step by step.\n\n"
     "Evaluate this polynomial: {data}\n\n"
     "Substitute the value and compute each term. "
     "Show your work, then write the result as \\boxed{{answer}}.")

_reg("number_base_conversion", "cot",
     "Think step by step.\n\n"
     "Convert this number between bases: {data}\n\n"
     "Use repeated division or positional notation. "
     "Show your work, then write the result as \\boxed{{answer}}.")

_reg("matrix_operations", "cot",
     "Think step by step.\n\n"
     "Perform this matrix operation: {data}\n\n"
     "Process element by element. "
     "Show your work, then write the result matrix as \\boxed{{answer}}.")

_reg("logical_operations", "cot",
     "Think step by step.\n\n"
     "Evaluate this logical expression: {data}\n\n"
     "Apply the logical operator to the operands. "
     "Show your reasoning, then write the result as \\boxed{{answer}}.")

_reg("combinatorics", "cot",
     "Think step by step.\n\n"
     "Solve this combinatorics problem: {data}\n\n"
     "Use the appropriate formula (C(n,r), P(n,r), or n!). "
     "Show your work, then write the result as \\boxed{{answer}}.")

_reg("pattern_completion", "cot",
     "Think step by step.\n\n"
     "Find the next element in this pattern: {data}\n\n"
     "Identify the underlying rule. "
     "Show your reasoning, then write the next element as \\boxed{{answer}}.")

_reg("algebraic_sequence", "cot",
     "Think step by step.\n\n"
     "Solve this algebraic sequence problem: {data}\n\n"
     "Identify the formula and apply it. "
     "Show your work, then write your answer as \\boxed{{answer}}.")

_reg("complex_pattern", "cot",
     "Think step by step.\n\n"
     "Analyze and continue this pattern: {data}\n\n"
     "Look for differences, ratios, or other relationships between terms. "
     "Show your reasoning, then write your answer as \\boxed{{answer}}.")

# --- HARD ---
_reg("shortest_path", "cot",
     "Think step by step.\n\n"
     "Find the shortest path in this graph: {data}\n\n"
     "Apply Dijkstra's algorithm or examine all paths. "
     "Show your reasoning, then write the shortest distance as \\boxed{{answer}}.")

_reg("knapsack", "cot",
     "Think step by step.\n\n"
     "Solve this knapsack problem: {data}\n\n"
     "Use dynamic programming. Consider each item and decide whether to include it. "
     "Show your reasoning, then write the maximum value as \\boxed{{answer}}.")

_reg("traveling_salesman", "cot",
     "Think step by step.\n\n"
     "Solve this traveling salesman problem: {data}\n\n"
     "Consider all possible routes and find the minimum total distance. "
     "Show your reasoning, then write the minimum distance as \\boxed{{answer}}.")

_reg("longest_common_subsequence", "cot",
     "Think step by step.\n\n"
     "Find the longest common subsequence: {data}\n\n"
     "Use dynamic programming. Build the LCS table. "
     "Show your reasoning, then write the LCS length as \\boxed{{answer}}.")

_reg("edit_distance", "cot",
     "Think step by step.\n\n"
     "Compute the edit (Levenshtein) distance: {data}\n\n"
     "Count the minimum insertions, deletions, and substitutions. "
     "Show your reasoning, then write the distance as \\boxed{{answer}}.")

_reg("coin_change", "cot",
     "Think step by step.\n\n"
     "Solve this coin change problem: {data}\n\n"
     "Use dynamic programming. Find the minimum coins needed. "
     "Show your reasoning, then write the count as \\boxed{{answer}}.")

_reg("interval_scheduling", "cot",
     "Think step by step.\n\n"
     "Solve this interval scheduling problem: {data}\n\n"
     "Sort by end time and greedily select non-overlapping intervals. "
     "Show your reasoning, then write the maximum count as \\boxed{{answer}}.")

_reg("topological_sort", "cot",
     "Think step by step.\n\n"
     "Find a valid topological ordering: {data}\n\n"
     "Process nodes with no incoming edges first. "
     "Show your reasoning, then write the ordering as \\boxed{{answer}}.")

_reg("regex_matching", "cot",
     "Think step by step.\n\n"
     "Determine if the string matches the regex pattern: {data}\n\n"
     "Walk through the pattern character by character. "
     "Show your reasoning, then write True or False as \\boxed{{answer}}.")

_reg("minimax_game", "cot",
     "Think step by step.\n\n"
     "Evaluate this game tree using minimax: {data}\n\n"
     "Alternate between maximizing and minimizing at each level. "
     "Show your reasoning, then write the optimal value as \\boxed{{answer}}.")

_reg("boolean_sat", "cot",
     "Think step by step.\n\n"
     "Evaluate this Boolean satisfiability problem: {data}\n\n"
     "Substitute the variable values and evaluate each clause. "
     "Show your reasoning, then write the result as \\boxed{{answer}}.")

_reg("n_queens", "cot",
     "Think step by step.\n\n"
     "Solve this N-Queens problem: {data}\n\n"
     "Place queens so that no two attack each other. "
     "Show your reasoning, then write your answer as \\boxed{{answer}}.")

_reg("sudoku_solving", "cot",
     "Think step by step.\n\n"
     "Solve this Sudoku puzzle: {data}\n\n"
     "Apply constraint propagation and backtracking. "
     "Show your reasoning, then write the solution as \\boxed{{answer}}.")

_reg("graph_coloring", "cot",
     "Think step by step.\n\n"
     "Find the minimum coloring for this graph: {data}\n\n"
     "Assign colors so no adjacent nodes share a color. "
     "Show your reasoning, then write your answer as \\boxed{{answer}}.")

_reg("tower_hanoi", "cot",
     "Think step by step.\n\n"
     "Solve this Tower of Hanoi problem: {data}\n\n"
     "Move disks one at a time, never placing a larger disk on a smaller one. "
     "Show your reasoning, then write your answer as \\boxed{{answer}}.")

_reg("matrix_chain_multiplication", "cot",
     "Think step by step.\n\n"
     "Find the optimal matrix chain multiplication order: {data}\n\n"
     "Use dynamic programming to minimize total scalar multiplications. "
     "Show your reasoning, then write the minimum cost as \\boxed{{answer}}.")

_reg("cryptarithmetic", "cot",
     "Think step by step.\n\n"
     "Solve this cryptarithmetic puzzle: {data}\n\n"
     "Each letter represents a unique digit. "
     "Show your reasoning, then write the solution as \\boxed{{answer}}.")

_reg("constraint_optimization", "cot",
     "Think step by step.\n\n"
     "Solve this constrained optimization problem: {data}\n\n"
     "Identify the feasible region and find the optimal point. "
     "Show your reasoning, then write the optimal value as \\boxed{{answer}}.")

_reg("logic_grid_puzzles", "cot",
     "Think step by step.\n\n"
     "Solve this logic grid puzzle: {data}\n\n"
     "Use process of elimination with the given clues. "
     "Show your reasoning, then write the solution as \\boxed{{answer}}.")

_reg("modular_systems", "cot",
     "Think step by step.\n\n"
     "Solve this modular arithmetic problem: {data}\n\n"
     "Apply modular arithmetic rules. "
     "Show your work, then write your answer as \\boxed{{answer}}.")
