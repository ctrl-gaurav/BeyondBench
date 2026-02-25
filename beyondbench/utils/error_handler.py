"""Error handling utilities."""

import logging
from typing import Any


class ErrorHandler:
    """Comprehensive error handling and classification."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def is_rate_limit_error(self, error: Any) -> bool:
        """Check if error is due to API rate limiting."""
        error_str = str(error).lower()
        return any(term in error_str for term in [
            "rate limit", "too many requests", "quota exceeded",
            "rate_limit_exceeded", "429"
        ])

    def is_memory_error(self, error: Any) -> bool:
        """Check if error is due to memory issues."""
        error_str = str(error).lower()
        return any(term in error_str for term in [
            "out of memory", "cuda out of memory", "oom",
            "memory error", "allocation failed"
        ])

    def is_timeout_error(self, error: Any) -> bool:
        """Check if error is due to timeout."""
        error_str = str(error).lower()
        return any(term in error_str for term in [
            "timeout", "timed out", "connection timeout"
        ])

    def get_error_category(self, error: Any) -> str:
        """Categorize error for better handling."""
        if self.is_rate_limit_error(error):
            return "rate_limit"
        elif self.is_memory_error(error):
            return "memory"
        elif self.is_timeout_error(error):
            return "timeout"
        else:
            return "unknown"