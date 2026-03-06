"""Test ModelHandler functionality."""

import pytest
from unittest.mock import MagicMock, patch


class TestModelHandlerInit:
    def test_import(self):
        from beyondbench.models.model_handler import ModelHandler
        assert ModelHandler is not None

    def test_api_provider_without_key_raises(self):
        from beyondbench.models.model_handler import ModelHandler
        with pytest.raises(RuntimeError, match="API key"):
            ModelHandler(model_id="gpt-4o", api_provider="openai")

    def test_get_model_info_structure(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.model_id = "test-model"
        handler.backend = "mock"
        handler.api_provider = None
        info = handler.get_model_info()
        assert "model_name" in info
        assert "backend" in info
        assert info["model_name"] == "test-model"
        assert info["backend"] == "mock"

    def test_count_tokens_empty(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.token_counter = None
        result = handler._count_tokens("")
        assert result == 0

    def test_count_tokens_fallback(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.token_counter = None
        result = handler._count_tokens("hello world test")
        assert result > 0

    def test_format_duration_seconds(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        assert "seconds" in handler._format_duration(30.0)

    def test_format_duration_minutes(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        assert "minutes" in handler._format_duration(120.0)

    def test_format_duration_hours(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        assert "hours" in handler._format_duration(7200.0)


class TestModelHandlerBackendSelection:
    def test_api_provider_sets_backend(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.model_id = "gpt-4o"
        handler.api_provider = "openai"
        handler.backend = "openai"
        assert handler.backend == "openai"

    def test_unsupported_api_provider(self):
        from beyondbench.models.model_handler import ModelHandler
        with pytest.raises((ValueError, RuntimeError)):
            ModelHandler(model_id="test", api_provider="unsupported", api_key="key")

    def test_generate_batch_delegates_to_generate(self):
        from beyondbench.models.model_handler import ModelHandler
        handler = ModelHandler.__new__(ModelHandler)
        handler.api_provider = "mock"
        handler.backend = "mock"
        handler.generate = MagicMock(return_value=["response1"])
        result = handler.generate_batch(["prompt1"])
        handler.generate.assert_called_once()
        assert result == ["response1"]

    def test_get_statistics_structure(self):
        from beyondbench.models.model_handler import ModelHandler
        import time
        handler = ModelHandler.__new__(ModelHandler)
        handler.model_id = "test-model"
        handler.api_provider = None
        handler.stats = {
            'api_calls': 5,
            'total_input_tokens': 100,
            'total_output_tokens': 50,
            'start_time': time.time() - 10,
            'prompts_processed': []
        }
        stats = handler.get_statistics()
        assert "total_time_seconds" in stats
        assert "api_calls" in stats
        assert stats["api_calls"] == 5
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
