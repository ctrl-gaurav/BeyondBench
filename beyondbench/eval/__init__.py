"""Evaluation utilities: baseline management and regression detection."""

from .baseline import BaselineManager
from .regression import RegressionChecker, run_regression_check

__all__ = ["BaselineManager", "RegressionChecker", "run_regression_check"]
