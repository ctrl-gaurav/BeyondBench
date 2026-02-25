"""
Pytest configuration and fixtures.
"""

import pytest
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def package_root():
    """Return package root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def beyondbench_module():
    """Import and return beyondbench module."""
    import beyondbench
    return beyondbench


@pytest.fixture
def sample_boxed_responses():
    """Sample boxed responses for testing parsing."""
    return [
        (r"\boxed{42}", 42),
        (r"The answer is \boxed{123}", 123),
        (r"\boxed{-15}", -15),
        (r"\boxed{3.14}", 3.14),
        (r"\boxed{[1, 2, 3]}", [1, 2, 3]),
    ]


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "Calculate the sum of [1, 2, 3, 4, 5]",
        "Find the maximum value in [10, 5, 8, 12, 3]",
        "Sort the following list: [5, 2, 8, 1, 9]",
    ]


@pytest.fixture
def mock_model_response():
    """Mock model response for testing."""
    return r"Let me solve this step by step. The answer is \boxed{42}."


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API keys"
    )
