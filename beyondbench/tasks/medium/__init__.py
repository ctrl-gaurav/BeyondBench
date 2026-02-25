"""Medium suite task implementations."""

# Import all task classes from the medium suite
try:
    from .algebraic_sequence_task import AlgebraicSequenceTask
    from .complex_pattern_task import ComplexPatternTask
    from .fibonacci_sequence_task import FibonacciSequenceTask
    from .geometric_sequence_task import GeometricSequenceTask
    from .prime_sequence_task import PrimeSequenceTask

except ImportError as e:
    print(f"Warning: Failed to import some medium task classes: {e}")

__all__ = [
    "AlgebraicSequenceTask",
    "ComplexPatternTask",
    "FibonacciSequenceTask",
    "GeometricSequenceTask",
    "PrimeSequenceTask"
]