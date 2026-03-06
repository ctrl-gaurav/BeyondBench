"""Pytest configuration and shared fixtures."""

import pytest
import sys
import os
import json
import tempfile
import shutil
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def package_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test results."""
    output_dir = tmp_path / "test_results"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def mock_model_handler():
    """Mock ModelHandler that returns canned responses with boxed format."""
    handler = MagicMock()
    handler.backend = "mock"
    handler.api_provider = None
    handler.model_id = "test-model"
    handler.tokenizer = None

    def mock_generate(prompts, **kwargs):
        responses = []
        for prompt in prompts:
            if "sort" in prompt.lower():
                responses.append(r"The sorted list is \boxed{[-5, 0, 3, 7, 12]}")
            elif "sum" in prompt.lower():
                responses.append(r"The sum is \boxed{42}")
            elif "maximum" in prompt.lower() or "max" in prompt.lower():
                responses.append(r"The maximum is \boxed{99}")
            elif "minimum" in prompt.lower() or "min" in prompt.lower():
                responses.append(r"The minimum is \boxed{-10}")
            elif "count" in prompt.lower():
                responses.append(r"The count is \boxed{5}")
            elif "median" in prompt.lower():
                responses.append(r"The median is \boxed{7.5}")
            elif "mean" in prompt.lower():
                responses.append(r"The mean is \boxed{15.3}")
            elif "mode" in prompt.lower():
                responses.append(r"The mode is \boxed{4}")
            elif "compare" in prompt.lower():
                responses.append(r"The first number is \boxed{greater than} the second")
            elif "sequence" in prompt.lower() or "next term" in prompt.lower():
                responses.append(r"The next term is \boxed{13}")
            else:
                responses.append(r"The answer is \boxed{42}")
        return responses

    handler.generate = MagicMock(side_effect=mock_generate)
    handler.generate_batch = handler.generate
    handler.get_model_info = MagicMock(return_value={
        "model_name": "test-model",
        "backend": "mock",
        "api_provider": None,
        "model_type": "mock"
    })
    handler.get_statistics = MagicMock(return_value={
        "total_time_seconds": 1.0,
        "api_calls": 10,
        "total_input_tokens": 1000,
        "total_output_tokens": 500,
    })
    handler.count_tokens = MagicMock(return_value=10)
    return handler


@pytest.fixture
def sample_boxed_responses():
    return [
        (r"\boxed{42}", 42),
        (r"The answer is \boxed{123}", 123),
        (r"\boxed{-15}", -15),
        (r"\boxed{3.14}", 3.14),
    ]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "api: marks tests that require API keys")
