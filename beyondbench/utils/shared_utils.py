"""
Shared utility functions for beyondbench package.

This module provides common utility functions used across different task types,
particularly for medium and hard suite tasks that require mathematical operations
and validation.
"""

import math
from typing import Union, Any


def values_are_close(val1: Union[int, float], val2: Union[int, float], tolerance: float = 1e-6) -> bool:
    """
    Check if two values are close within a specified tolerance.

    Args:
        val1: First value
        val2: Second value
        tolerance: Tolerance for comparison (default: 1e-6)

    Returns:
        bool: True if values are within tolerance, False otherwise
    """
    if val1 is None or val2 is None:
        return val1 == val2

    try:
        val1_num = float(val1)
        val2_num = float(val2)
        return abs(val1_num - val2_num) <= tolerance
    except (ValueError, TypeError):
        return str(val1) == str(val2)


def round_if_close_to_int(value: Union[int, float], tolerance: float = 1e-10) -> Union[int, float]:
    """
    Round a float to an integer if it's very close to an integer value.

    Args:
        value: The value to potentially round
        tolerance: How close to an integer the value must be (default: 1e-10)

    Returns:
        int if close to integer, otherwise the original float
    """
    if not isinstance(value, (int, float)):
        return value

    if isinstance(value, int):
        return value

    rounded = round(value)
    if abs(value - rounded) <= tolerance:
        return rounded
    return value


def is_valid_number(value: Any) -> bool:
    """
    Check if a value is a valid number (int or float).

    Args:
        value: Value to check

    Returns:
        bool: True if value is a valid number, False otherwise
    """
    if isinstance(value, (int, float)):
        return not (math.isnan(value) if isinstance(value, float) else False)

    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False

    return False


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero.

    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Default value to return if division by zero (default: 0.0)

    Returns:
        float: Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except (ValueError, TypeError):
        return default


def clamp(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Union[int, float]:
    """
    Clamp a value between minimum and maximum bounds.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def calculate_percentage(part: Union[int, float], whole: Union[int, float]) -> float:
    """
    Calculate percentage safely.

    Args:
        part: Part value
        whole: Whole value

    Returns:
        float: Percentage (0-100)
    """
    if whole == 0:
        return 0.0
    return (float(part) / float(whole)) * 100


def format_number(value: Union[int, float], precision: int = 3) -> str:
    """
    Format a number for display with appropriate precision.

    Args:
        value: Number to format
        precision: Number of decimal places for floats

    Returns:
        str: Formatted number string
    """
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        # Round to integer if very close
        rounded = round_if_close_to_int(value)
        if isinstance(rounded, int):
            return str(rounded)
        return f"{value:.{precision}f}"
    else:
        return str(value)


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.

    Args:
        n: Number to check

    Returns:
        bool: True if n is a power of two, False otherwise
    """
    return n > 0 and (n & (n - 1)) == 0


def gcd(a: int, b: int) -> int:
    """
    Calculate greatest common divisor using Euclidean algorithm.

    Args:
        a: First number
        b: Second number

    Returns:
        int: Greatest common divisor
    """
    while b:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    """
    Calculate least common multiple.

    Args:
        a: First number
        b: Second number

    Returns:
        int: Least common multiple
    """
    return abs(a * b) // gcd(a, b) if a and b else 0


def fibonacci(n: int) -> int:
    """
    Calculate nth Fibonacci number efficiently.

    Args:
        n: Position in Fibonacci sequence (0-indexed)

    Returns:
        int: nth Fibonacci number
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_prime(n: int) -> bool:
    """
    Check if a number is prime.

    Args:
        n: Number to check

    Returns:
        bool: True if prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def factorial(n: int) -> int:
    """
    Calculate factorial of n.

    Args:
        n: Number to calculate factorial for

    Returns:
        int: n! (factorial of n)
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i
    return result